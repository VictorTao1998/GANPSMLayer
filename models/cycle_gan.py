"""
Author: Isabella Liu 8/13/21
Feature: Cycle GAN Model
Reference: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py
"""

import torch
import itertools
from .gan_networks import define_G, define_D, GANLoss
from utils.image_pool import ImagePool
#from utils.test_util import load_from_dataparallel_model


class CycleGANModel:
    def __init__(self, device, lambdaA=10.0, lambdaB=10.0, lambda_identity=0.5, isTrain=True):
        """
        lambdaA: weight for cycle loss (A -> B -> A)
        lambdaB: weight for cycle loss (B -> A -> B)
        lambda_identity: use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight
            of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller
            than the weight of the reconstruction loss, please set lambda_identity = 0.1
        """
        self.lambda_A = lambdaA
        self.lambda_B = lambdaB
        self.lambda_idt = lambda_identity
        self.isTrain = isTrain
        self.device = device

        # Define networks for both generators and discriminators
        self.netG_A = define_G(input_nc=1, output_nc=1, ngf=64, netG='resnet_9blocks', norm='instance')
        self.netG_B = define_G(input_nc=1, output_nc=1, ngf=64, netG='resnet_9blocks', norm='instance')

        if self.isTrain:
            self.netD_A = define_D(input_nc=1, ndf=64, netD='basic')
            self.netD_B = define_D(input_nc=1, ndf=64, netD='basic')
            # Create image buffer to store previously generated images
            self.fake_A_pool = ImagePool(pool_size=50)
            self.fake_B_pool = ImagePool(pool_size=50)
            # Define loss functions
            self.criterionGAN = GANLoss(gan_mode='lsgan')
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # Initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=0.0002, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=0.0002, betas=(0.5, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_device(self, device):
        for net in [self.netG_A, self.netG_B, self.netD_A, self.netD_B, self.criterionGAN]:
            net = net.to(device)

    def set_distributed(self, is_distributed, local_rank):
        """Set distributed training"""
        for net in [self.netG_A, self.netG_B, self.netD_A, self.netD_B]:
            if is_distributed:
                net = torch.nn.parallel.DistributedDataParallel(
                    net, device_ids=[local_rank], output_device=local_rank
                )
            else:
                net = torch.nn.DataParallel(net)

    def load_model(self, file_name):
        G_A_dict = torch.load(file_name)['G_A']
        G_B_dict = torch.load(file_name)['G_B']
        self.netG_A.load_state_dict(G_A_dict)
        self.netG_B.load_state_dict(G_B_dict)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_input(self, simfeaL, simfeaR, realfeaL, realfeaR, real_gt):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        #print(simfeaL.shape)
        self.sim_A_L = simfeaL.to(self.device)[:,1,:,:].reshape((simfeaL.shape[0], 1, simfeaL.shape[2], simfeaL.shape[3]))
        self.sim_A_R = simfeaR.to(self.device)[:,1,:,:].reshape((simfeaL.shape[0], 1, simfeaL.shape[2], simfeaL.shape[3]))
        self.real_A_L = realfeaL.to(self.device)[:,1,:,:].reshape((simfeaL.shape[0], 1, simfeaL.shape[2], simfeaL.shape[3]))
        self.real_A_R = realfeaR.to(self.device)[:,1,:,:].reshape((simfeaL.shape[0], 1, simfeaL.shape[2], simfeaL.shape[3]))

    def forward(self):
        # fake_B = G_A(A)
        self.fake_B_L = self.netG_A(self.real_A_L)  # G_A(A)
        self.rec_A_L = self.netG_B(self.fake_B_L)   # G_B(G_A(A))
        self.fake_A_L = self.netG_B(self.sim_A_L)  # G_B(B)
        self.rec_B_L = self.netG_A(self.fake_A_L)   # G_A(G_B(B))

        self.fake_B_R = self.netG_A(self.real_A_R)  # G_A(A)
        self.rec_A_R = self.netG_B(self.fake_B_R)   # G_B(G_A(A))
        self.fake_A_R = self.netG_B(self.sim_A_R)  # G_B(B)
        self.rec_B_R = self.netG_A(self.fake_A_R)   # G_A(G_B(B))

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        print(self.lambda_identity, self.lambda_A, self.lambda_B)
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A_L = self.netG_A(self.sim_A_L)
            self.loss_idt_A_L = self.criterionIdt(self.idt_A_L, self.sim_A_L) * self.lambda_B * self.lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B_L = self.netG_B(self.real_A_L)
            self.loss_idt_B_L = self.criterionIdt(self.idt_B_L, self.real_A_L) * self.lambda_A * self.lambda_idt

            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A_R = self.netG_A(self.sim_A_R)
            self.loss_idt_A_R = self.criterionIdt(self.idt_A_R, self.sim_A_R) * self.lambda_B * self.lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B_R = self.netG_B(self.real_A_R)
            self.loss_idt_B_R = self.criterionIdt(self.idt_B_R, self.real_A_R) * self.lambda_A * self.lambda_idt
        else:
            self.loss_idt_A_L = 0
            self.loss_idt_B_L = 0
            self.loss_idt_A_R = 0
            self.loss_idt_B_R = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A_L = self.criterionGAN(self.netD_A(self.fake_B_L), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B_L = self.criterionGAN(self.netD_B(self.fake_A_L), True)

        # GAN loss D_A(G_A(A))
        self.loss_G_A_R = self.criterionGAN(self.netD_A(self.fake_B_R), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B_R = self.criterionGAN(self.netD_B(self.fake_A_R), True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A_L = self.criterionCycle(self.rec_A_L, self.real_A_L) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B_L = self.criterionCycle(self.rec_B_L, self.sim_A_L) * lambda_B

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A_R = self.criterionCycle(self.rec_A_R, self.real_A_R) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B_R = self.criterionCycle(self.rec_B_R, self.sim_A_R) * lambda_B

        # combined loss and calculate gradients
        self.loss_G = (self.loss_G_A_L + self.loss_G_B_L + self.loss_G_A_R + self.loss_G_B_R) * 0.5 + \
                        (self.loss_cycle_A_L + self.loss_cycle_B_L + self.loss_cycle_A_R + self.loss_cycle_B_R) * 0.5 + \
                        (self.loss_idt_A_L + self.loss_idt_B_L + self.loss_idt_A_R + self.loss_idt_B_R) * 0.5
        self.loss_G.backward()

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        #loss_D.backward()
        #del pred_real, loss_D_real, pred_fake
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B_L = self.fake_B_pool.query(self.fake_B_L)
        self.loss_D_A_L = self.backward_D_basic(self.netD_A, self.sim_A_L, fake_B_L)

        fake_B_R = self.fake_B_pool.query(self.fake_B_R)
        self.loss_D_A_R = self.backward_D_basic(self.netD_A, self.sim_A_R, fake_B_R)

        self.loss_D_A = (self.loss_D_A_L + self.loss_D_A_R) * 0.5
        self.loss_D_A.backward()
        #del fake_B_L, fake_B_R

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A_L = self.fake_B_pool.query(self.fake_A_L)
        self.loss_D_B_L = self.backward_D_basic(self.netD_B, self.real_A_L, fake_A_L)

        fake_A_R = self.fake_B_pool.query(self.fake_A_R)
        self.loss_D_B_R = self.backward_D_basic(self.netD_B, self.real_A_R, fake_A_R)

        self.loss_D_B = (self.loss_D_B_L + self.loss_D_B_R) * 0.5
        self.loss_D_B.backward()
        #del fake_A_L, fake_A_R


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        #self.optimizer_psm.zero_grad()
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        #self.optimizer_psm.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def train(self):
        """Make models train mode during train time"""
        for net in [self.netG_A, self.netG_B, self.netD_A, self.netD_B]:
            net.train()

    def eval(self):
        """Make models eval mode during test time"""
        for net in [self.netG_A, self.netG_B, self.netD_A, self.netD_B]:
            net.eval()