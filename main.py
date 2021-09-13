from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader.messy_dataset import MESSYDataset
from dataloader.warp_ops import apply_disparity_cu
from models import *
from utils import *
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
#parser.add_argument('--datapath', default='/media/jiaren/ImageNet/SceneFlowData/',
#                    help='datapath')

parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--savemodel', default='/cephfs/jianyu',
                    help='save model')
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

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=50, help='the frequency of saving summary')
parser.add_argument('--test_summary_freq', type=int, default=50, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

parser.add_argument('--log_freq', type=int, default=50, help='log freq')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')
parser.add_argument("--local_rank", type=int, default=0)

parser.add_argument('--mode', type=str, default="train", help='train or test mode')


parser.add_argument('--ndisps', type=str, default="48,24", help='ndisps')
parser.add_argument('--disp_inter_r', type=str, default="4,1", help='disp_intervals_ratio')
parser.add_argument('--dlossw', type=str, default="0.5,2.0", help='depth loss weight for different stage')
parser.add_argument('--cr_base_chs', type=str, default="32,32,16", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='predicted disp detach, undetach')


parser.add_argument('--using_ns', action='store_true', help='using neighbor search')
parser.add_argument('--ns_size', type=int, default=3, help='nb_size')

parser.add_argument('--crop_height', type=int, required=True, help="crop height")
parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--test_crop_height', type=int, required=True, help="crop height")
parser.add_argument('--test_crop_width', type=int, required=True, help="crop width")

parser.add_argument('--using_apex', action='store_true', help='using apex, need to install apex')
parser.add_argument('--sync_bn', action='store_true',help='enabling apex sync BN.')
parser.add_argument('--opt-level', type=str, default="O0")
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)

parser.add_argument('--ground', action='store_true', help='include ground pixel')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

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

TrainImgLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=8, drop_last=True)

TestImgLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                            shuffle=False, num_workers=4, drop_last=False)

#TrainImgLoader = torch.utils.data.DataLoader(
#         DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
#         batch_size= 12, shuffle= True, num_workers= 8, drop_last=False)

#TestImgLoader = torch.utils.data.DataLoader(
#         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
#         batch_size= 8, shuffle= False, num_workers= 4, drop_last=False)


if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

def train(imgL,imgR, disp_L):
    model.train()

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    disp_gt_t = disp_L.reshape((args.batch_size,1,args.crop_height,args.crop_width))
    disparity_L_from_R = apply_disparity_cu(disp_gt_t, disp_gt_t.int())
    #disp_gt = disparity_L_from_R.reshape((1,2,256,512))
    disp_L = disparity_L_from_R.reshape((args.batch_size,args.crop_height,args.crop_width))


#---------
    mask = (disp_true < args.maxdisp) & (disp_true > 0)
    mask.detach_()
    #----
    optimizer.zero_grad()
    
    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL,imgR)
        output1 = torch.squeeze(output1,1)
        output2 = torch.squeeze(output2,1)
        output3 = torch.squeeze(output3,1)
        loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
    elif args.model == 'basic':
        output = model(imgL,imgR)
        output = torch.squeeze(output,1)
        loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

    image_outputs = {"disp_est": output, "disp_gt": disp_true, "imgL": imgL, "imgR": imgR}
    scalar_outputs = {}

    image_outputs["errormap"] = [disp_error_image_func.apply(output, disp_true)]
    scalar_outputs["loss"] = [loss.item()]


    loss.backward()
    optimizer.step()

    return loss.data, image_outputs, scalar_outputs

def test(imgL,imgR,disp_true):

    model.eval()

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

    disp_gt_t = disp_true.reshape((args.test_batch_size,1,args.test_crop_height,args.test_crop_width))
    disparity_L_from_R = apply_disparity_cu(disp_gt_t, disp_gt_t.int())
    #disp_gt = disparity_L_from_R.reshape((1,2,256,512))
    disp_true = disparity_L_from_R.reshape((args.test_batch_size,args.test_crop_height,args.test_crop_width))

    #---------
    mask = (disp_true < 192) & (disp_true > 0) 
    #----

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16       
        top_pad = (times+1)*16 -imgL.shape[2]
    else:
        top_pad = 0

    if imgL.shape[3] % 16 != 0:
        times = imgL.shape[3]//16                       
        right_pad = (times+1)*16-imgL.shape[3]
    else:
        right_pad = 0  

    imgL = F.pad(imgL,(0,right_pad, top_pad,0))
    imgR = F.pad(imgR,(0,right_pad, top_pad,0))

    with torch.no_grad():
        output3 = model(imgL,imgR)
        output3 = torch.squeeze(output3)
    
    if top_pad !=0:
        img = output3[:,top_pad:,:]
    else:
        img = output3

    if len(disp_true[mask])==0:
        loss = 0
    else:
        loss = F.l1_loss(img[mask],disp_true[mask]) #torch.mean(torch.abs(img[mask]-disp_true[mask]))  # end-point-error

    image_outputs = {"disp_est": img, "disp_gt": disp_true, "imgL": imgL, "imgR": imgR}
    scalar_outputs = {}

    image_outputs["errormap"] = [disp_error_image_func.apply(img, disp_true)]
    scalar_outputs["loss"] = [loss.item()]

    return loss.data.cpu(), image_outputs, scalar_outputs

def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    logger = SummaryWriter(args.logdir)

    start_full_time = time.time()
    for epoch in range(0, args.epochs):
        print('This is %d-th epoch' %(epoch))
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch)

        ## training ##
        for batch_idx, data in enumerate(TrainImgLoader):
                imgL_crop, imgR_crop, disp_crop_L = data['left'], data['right'], data['disparity']

                global_step = len(TrainImgLoader) * epoch + batch_idx
                do_summary = global_step % args.summary_freq == 0
                start_time = time.time()

                loss, image_outputs, scalar_outputs = train(imgL_crop,imgR_crop, disp_crop_L)

                if do_summary:
                    save_scalars(logger, 'train', scalar_outputs, global_step)
                    save_images(logger, 'train', image_outputs, global_step)
                    #save_texts(logger, 'train', text_outputs, global_step)
                del scalar_outputs, image_outputs

                print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
                total_train_loss += loss
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))

        #SAVE
        savefilename = args.logdir+'/checkpoint_'+str(epoch)+'.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss/len(TrainImgLoader),
        }, savefilename)

    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))

    #------------- TEST ------------------------------------------------------------
    total_test_loss = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        global_step = len(TestImgLoader) * epoch + batch_idx
        do_summary = global_step % args.test_summary_freq == 0
        test_loss, image_outputs, scalar_outputs = test(imgL,imgR, disp_L)

        if do_summary:
            save_scalars(logger, 'test', scalar_outputs, global_step)
            save_images(logger, 'test', image_outputs, global_step)
            #save_texts(logger, 'train', text_outputs, global_step)
        del scalar_outputs, image_outputs
        
        print('Iter %d test loss = %.3f' %(batch_idx, test_loss))
        total_test_loss += test_loss

    print('total test loss = %.3f' %(total_test_loss/len(TestImgLoader)))
    #----------------------------------------------------------------------------------
    #SAVE test information
    savefilename = args.logdir+'testinformation.tar'
    torch.save({
            'test_loss': total_test_loss/len(TestImgLoader),
        }, savefilename)


if __name__ == '__main__':
    main()
    
