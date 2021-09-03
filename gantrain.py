from __future__ import print_function, division
import argparse
import os, sys, shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from dataloader.messy_dataset import MESSYDataset

from models import *

from utils import *
from utils.visualization import *
from utils.experiment import *
import gc
from dataloader.warp_ops import *


import torchvision.transforms as transforms
from options.train_options import TrainOptions
from models.cycle_gan_model import *
from models import create_model
from dataloader.warp_ops import apply_disparity_cu


cudnn.benchmark = True
assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

parser = argparse.ArgumentParser(description='PSMNet')

parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--cmodel', default='stackhourglass',
                    help='select model')
#parser.add_argument('--datapath', default='/media/jiaren/ImageNet/SceneFlowData/',
#                    help='datapath')

parser.add_argument('--loadmodel', default= None,
                    help='load model')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')


parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--depthpath', required=True, help='depth path')
parser.add_argument('--test_datapath', required=True, help='data path')
parser.add_argument('--test_sim_datapath', required=True, help='data path')
parser.add_argument('--test_real_datapath', required=True, help='data path')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--sim_testlist', required=True, help='testing list')
parser.add_argument('--real_testlist', required=True, help='testing list')

parser.add_argument('--clr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--cbatch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')

#parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=50, help='the frequency of saving summary')
parser.add_argument('--test_summary_freq', type=int, default=50, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

parser.add_argument('--log_freq', type=int, default=50, help='log freq')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')
parser.add_argument("--local_rank", type=int, default=0)

parser.add_argument('--using_ns', action='store_true', help='using neighbor search')
parser.add_argument('--ns_size', type=int, default=3, help='nb_size')

parser.add_argument('--crop_height', type=int, required=True, help="crop height")
parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--test_crop_height', type=int, required=True, help="crop height")
parser.add_argument('--test_crop_width', type=int, required=True, help="crop width")


parser.add_argument('--ground', action='store_true', help='include ground pixel')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



# parse arguments
#args = parser.parse_args()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)

train_dataset = MESSYDataset(args.datapath, args.depthpath, args.trainlist, True,
                              crop_height=args.crop_height, crop_width=args.crop_width,
                              test_crop_height=args.test_crop_height, test_crop_width=args.test_crop_width,
                              left_img="0128_irL_denoised_half.png", right_img="0128_irR_denoised_half.png", args=args)

test_dataset = MESSYDataset(args.test_datapath, args.depthpath, args.testlist, False,
                             crop_height=args.crop_height, crop_width=args.crop_width,
                             test_crop_height=args.test_crop_height, test_crop_width=args.test_crop_width,
                             left_img="0128_irL_denoised_half.png", right_img="0128_irR_denoised_half.png", args=args)

test_real_dataset = MESSYDataset(args.test_real_datapath, args.depthpath, args.real_testlist, True,
                             crop_height=args.crop_height, crop_width=args.crop_width,
                             test_crop_height=args.test_crop_height, test_crop_width=args.test_crop_width,
                             left_img="1024_irL_real_1080.png", right_img="1024_irR_real_1080.png", args=args)

real_sampler = torch.utils.data.RandomSampler(test_real_dataset)

TrainImgLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.cbatch_size,
                                                 shuffle=True, num_workers=4, drop_last=True)

TestImgLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                            shuffle=False, num_workers=4, drop_last=False)

RealImgLoader = torch.utils.data.DataLoader(test_real_dataset, batch_size=args.cbatch_size, sampler=real_sampler,
                                                shuffle=False, num_workers=4, drop_last=False)

logger = SummaryWriter(args.logdir)

#TrainImgLoader = torch.utils.data.DataLoader(
#         DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
#         batch_size= 12, shuffle= True, num_workers= 8, drop_last=False)

#TestImgLoader = torch.utils.data.DataLoader(
#         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
#         batch_size= 8, shuffle= False, num_workers= 4, drop_last=False)


if args.cmodel == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.cmodel == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    print('need pretrain')

print('Load pretrained model')
pretrain_dict = torch.load(args.loadmodel)
model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

#discriminator = Discriminator(1, args.feat_map).cuda()
#discriminator.apply(weights_init)

feaex = model.module.feature_extraction.ganfeature

opt = TrainOptions().parse()

#opt_s1.input_nc = 32
#opt_s1.output_nc = 32

#opt_s2.input_nc = 16
#opt_s2.output_nc = 16

#opt_s1.checkpoints_dir = args.logdir
opt.checkpoints_dir = args.logdir

#print(opt_s1.model, opt_s2.model)


#s1_gan  = create_model(opt_s1)      # create a model given opt.model and other options
#s1_gan.setup(opt_s1)

      # create a model given opt.model and other options
#s2_gan.setup(opt_s2)



optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

model.module.feature_extraction.gan_train = True

c_gan = create_model(opt, model)
c_gan.setup(opt)

start_epoch = 0

def main():
    avg_test_scalars = None
    Cur_D1 = 1
    #model.set_gan_train()
    feaex.eval()
    for epoch_idx in range(start_epoch, args.epochs):
        #adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)
        c_gan.update_learning_rate()

        # training
        for batch_idx, simsample in enumerate(TrainImgLoader):
            print(batch_idx)
            realsample = next(iter(RealImgLoader))
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            #loss, scalar_outputs, image_outputs = train_sample(sample, batch_idx, compute_metrics=do_summary)
            simfeaL, simfeaR, sim_gt = feaex(simsample['left'].cuda()), feaex(simsample['right'].cuda()), simsample['disparity'].cuda()
            realfeaL, realfeaR, real_gt = feaex(realsample['left'].cuda()), feaex(realsample['right'].cuda()), realsample['disparity'].cuda()
            #real_gt = real_gt.reshape((args.cbatch_size,1,args.crop_height,args.crop_width))

            disp_gt_t = real_gt.reshape((args.cbatch_size,1,args.crop_height,args.crop_width))
            real_gt = apply_disparity_cu(disp_gt_t, disp_gt_t.int())

            #disp_gt = disparity_L_from_R.reshape((1,2,256,512))
            #real_gt = disparity_L_from_R.reshape((args.cbatch_size,args.crop_height,args.crop_width))

            #print("sim_fea: ", simfeaL['stage1'].shape, simfeaL['stage2'].shape)

            c_gan.set_input(simfeaL.detach(), simfeaR.detach(), realfeaL.detach(), realfeaR.detach(), real_gt)         # unpack data from dataset and apply preprocessing
            c_gan.optimize_parameters()

            

            if batch_idx % 50 == 0:
                feature_outputs_sim = [simfeaL[:,i,:,:] for i in range(3)]
                feature_outputs_real = [realfeaL[:,i,:,:] for i in range(3)]

                fakeSim = c_gan.fake_B_L

                feature_fake_sim = [fakeSim[:,i,:,:] for i in range(3)]

                outputs_1, outputs_2, outputs_3 = c_gan.psm_outputs0, c_gan.psm_outputs1, c_gan.psm_outputs2
                
                disp_ests = [outputs_1, outputs_2, outputs_3]
                image_outputs = {"imgSim": simsample['left'], "imgReal_L": realsample['left'], "imgReal_R": realsample['right'], "Dis_gt": real_gt, \
                            "Dis_est": disp_ests, "feature_sim": feature_outputs_sim, "feature_real": feature_outputs_real, "feature_fake_sim": feature_fake_sim}

                image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, realsample['disparity']) for disp_est in disp_ests]

                scalar_outputs = {"loss_G": c_gan.loss_G, "loss_D_A": c_gan.loss_D_A, "loss_D_B": c_gan.loss_D_B, "loss_G_A": (c_gan.loss_G_A_L + c_gan.loss_G_A_R) * 0.5, \
                                "loss_G_B": (c_gan.loss_G_B_L + c_gan.loss_G_B_R) * 0.5}


                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)

        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:

            #print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            c_gan.save_networks('latest')
            c_gan.save_networks(epoch_idx)

            c_gan.save_psm(epoch_idx)

        gc.collect()

            # saving bset checkpoints



if __name__ == '__main__':
    main()