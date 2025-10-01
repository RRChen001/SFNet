import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
import cv2
import numpy as np
import math
import lpips
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
from .base_model import BaseModel
from . import networks
from .VGG16 import VGG16
from skimage.metrics import structural_similarity as ssim
import functools
from models import networks
from models import HSV
from skimage.color import rgb2gray
from skimage.feature import canny


class FrequencyLoss(nn.Module):
    """
    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=True):
        super(FrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    # def tensor2freq(self, x):
    #     # crop image patches
    #     patch_factor = self.patch_factor
    #     _, _, h, w = x.shape
    #     assert h % patch_factor == 0 and w % patch_factor == 0, (
    #         'Patch factor should be divisible by image height and width')
    #     patch_list = []
    #     patch_h = h // patch_factor
    #     patch_w = w // patch_factor
    #     for i in range(patch_factor):
    #         for j in range(patch_factor):
    #             patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

    #     # stack to patch tensor
    #     y = torch.stack(patch_list, 1)

    #     # perform 2D DFT (real-to-complex, orthonormalization)
    #     if IS_HIGH_VERSION:
    #         freq = torch.fft.fft2(y, norm='ortho')
    #         freq = torch.stack([freq.real, freq.imag], -1)
    #     else:
    #         freq = torch.rfft(y, 2, onesided=False, normalized=True)
    #     return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        # print('1111111111111111111')
        # print(torch.mean(loss).shape)
        # print('1111111111111111111')
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = pred
        target_freq = target

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight



class FFTProcessor:
    def __init__(self, img_col):
        self.img_col = img_col

    def fft_compute_color_batch(self, center=True):
        assert self.img_col.size(1) == 3, "Should be color image"
        batch_size, _, h, w = self.img_col.size()
        lims_list = []
        idx_list_ = []
        x_mag = torch.zeros((batch_size, 3, h, w))
        x_phase = torch.zeros((batch_size, 3, h, w))
        x_fft = torch.zeros((batch_size, 6, h, w))

        for i in range(batch_size):
            for j in range(3):
                img = self.img_col[i, j]
                dft = torch.fft.fft2(img, dim=(-2, -1))
                if center:
                    dft = torch.fft.fftshift(dft, dim=(-2, -1))
                mag = torch.abs(dft)
                idx = (mag == 0)
                mag[idx] = 1.
                magnitude_spectrum = torch.log(mag)
                phase_spectrum = torch.angle(dft)
                x_mag[i, j] = magnitude_spectrum
                x_phase[i, j] = phase_spectrum

                x_fft[i, 2*j] = dft.real
                x_fft[i, 2*j+1] = dft.imag

                idx_list_.append(idx)
                # print('222222222222222222222222')
                # print(x_fft.shape)                
                # print('222222222222222222222222')
        return x_fft

class FFTProcessor_mag:
    def __init__(self, img_col):
        self.img_col = img_col

    def fft_compute_color_batch(self, center=True):
        assert self.img_col.size(1) == 3, "Should be color image"
        batch_size, _, h, w = self.img_col.size()
        lims_list = []
        idx_list_ = []
        x_mag = torch.zeros((batch_size, 3, h, w))
        x_phase = torch.zeros((batch_size, 3, h, w))
        x_fft = torch.zeros((batch_size, 6, h, w))

        for i in range(batch_size):
            for j in range(3):
                img = self.img_col[i, j]
                dft = torch.fft.fft2(img, dim=(-2, -1))
                if center:
                    dft = torch.fft.fftshift(dft, dim=(-2, -1))
                mag = torch.abs(dft)
                idx = (mag == 0)
                mag[idx] = 1.
                magnitude_spectrum = torch.log(mag)
                phase_spectrum = torch.angle(dft)
                x_mag[i, j] = magnitude_spectrum
                x_phase[i, j] = phase_spectrum

                x_fft[i, 2*j] = dft.real
                x_fft[i, 2*j+1] = dft.imag

                idx_list_.append(idx)
                # print('222222222222222222222222')
                # print(x_fft.shape)                
                # print('222222222222222222222222')
        return x_mag

class Ready(BaseModel):
    def name(self):
        return 'SFModel'

    def Edge_MSE_loss(Original_img, pred_img, Edge):

        loss = F.mse_loss(Original_img, pred_img, reduction='none')
        loss = loss * (1 - Edge) + loss * Edge * 10
        return loss.mean()

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.device = torch.device('cuda:1')
        self.opt = opt
        self.isTrain = opt.isTrain


        self.vgg=VGG16(requires_grad=False)       #将VGG16的4_3层之前的每个卷积组输出放在合适的顺次
        self.vgg=self.vgg.cuda()
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)        #batchSize=1，input_nc=3，fineSize=256, B,C,H,W
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,      #output_nc=3
                                   opt.fineSize, opt.fineSize)
        self.input_f = self.Tensor(opt.batchSize, opt.input_nc*2,
                                   opt.fineSize, opt.fineSize)
        self.img_gray_real = self.Tensor(opt.batchSize, opt.output_nc,    
                                   opt.fineSize, opt.fineSize)
        self.img_gray_Syn = self.Tensor(opt.batchSize, opt.output_nc,    
                                   opt.fineSize, opt.fineSize)                          
        #self.shape = np.shape
        # batchsize should be 1 for mask_global
        self.mask_global = torch.ByteTensor(1, 1, opt.fineSize, opt.fineSize)


        self.mask_global.zero_()
        self.mask_global[:, :, int(self.opt.fineSize/4) + self.opt.overlap : int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap,\
                                int(self.opt.fineSize/4) + self.opt.overlap: int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap] = 1

        self.mask_type = opt.mask_type
        self.gMask_opts = {}

        if len(opt.gpu_ids) > 0:
            self.use_gpu = True
            self.mask_global = self.mask_global.cuda()
        #device_ids = [i for i in range(torch.cuda.device_count())]
        device_ids = [1]
        
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.norm, opt.use_dropout, opt.init_type, device_ids, opt.init_gain)
        if len(device_ids)>1:
            self.netG = nn.DataParallel(self.netG, device_ids=device_ids)

        if self.isTrain:
            use_sigmoid = False
            if opt.gan_type == 'vanilla':
                use_sigmoid = True  # only vanilla GAN using BCECriterion

            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.norm, use_sigmoid, opt.init_type, device_ids, opt.init_gain)   #patch鉴别器
            #self.netD = nn.DataParallel(self.netD, device_ids=device_ids)
        if not self.isTrain or opt.continue_train:
            print('Loading pre-trained network!')
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_type=opt.gan_type, tensor=self.Tensor)   #GAN损失
            self.criterionL1 = torch.nn.L1Loss()                                              #L1距离
            self.criterionMSE = torch.nn.MSELoss()
            self.frequencyLoss = FrequencyLoss()
            #self.hsv = HSV.HSV()
            #self.edge = Edge_MSE_loss()
            #self.floss = self.frequencyLoss.loss_formulation(pred, target, matrix=None)

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))    #重新调整学习率

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)      #打印参数数量
            if self.isTrain:
                networks.print_network(self.netD)
            print('-----------------------------------------------')

        
    def set_input(self,input,mask):

        inputBatch, inputChannel, inputHeight, inputWeight = input.size()
        input_A = input
        input_B = input.clone()
        
        input_mask=mask

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

        self.image_paths = 0

        if self.opt.mask_type == 'center':
            self.mask_global=self.mask_global

        elif self.opt.mask_type == 'random':
            self.mask_global.zero_()
            self.mask_global=input_mask
        else:
            raise ValueError("Mask_type [%s] not recognized." % self.opt.mask_type)

        self.ex_mask = self.mask_global.expand(1, 3, self.mask_global.size(2), self.mask_global.size(3)) # 变成1*3*h*w

        self.inv_ex_mask = torch.add(torch.neg(self.ex_mask.float()), 1).byte()    #取反加1
        self.input_A.narrow(1,0,1).masked_fill_(self.mask_global, 2*123.0/255.0 - 1.0)   #取RGB第一个通道加mask
        self.input_A.narrow(1,1,1).masked_fill_(self.mask_global, 2*104.0/255.0 - 1.0)
        self.input_A.narrow(1,2,1).masked_fill_(self.mask_global, 2*117.0/255.0 - 1.0)

        

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def load_edge(self, img):
        return canny(img, sigma=2, mask=None).astype(np.float)

    def forward(self):
        self.netG.train()
        # self.real_A = self.input_A.to(self.device)
        # self.real_B = self.input_B.to(self.device)
        self.real_A = self.input_A.cuda()
        self.real_B = self.input_B.cuda()
        self.fft = FFTProcessor(self.real_B)
        self.input_f =self.fft.fft_compute_color_batch()
        self.input_f = self.input_f.cuda()


        
        #self.real_f = self.input_f.to(self.device)  # 真实图像利用cuda
        self.Syn, self.freqfake, self.edgeSyn = self.netG(self.real_A)

        self.img_gray_Syn = self.img_gray_Syn.cpu()
        # self.img_gray_Syn = rgb22gray(self.Syn) #函数名字好像是rgb2gray，多写了个2
        self.img_gray_Syn = rgb2gray(self.Syn)  # 函数名字好像是rgb2gray，多写了个2
        self.edge_Syn = self.load_edge(self.img_gray_Syn) 
        #self.Syn = self.netG(self.real_A)   #第一个子网的输出
        # self.un=self.fake_G.clone()
        # self.Unknowregion=self.un.data.masked_fill_(self.inv_ex_mask, 0)
        # self.knownregion=self.real_A.data.masked_fill_(self.ex_mask, 0)
        # self.Syn=self.Unknowregion+self.knownregion


    def test(self):
        self.netG.eval()
        self.real_A = self.input_A.cuda()#to(self.device)
        self.real_B = self.input_B.cuda()#to(self.device)
        self.real_f = self.input_f.cuda()#to(self.device)  # 真实图像利用cuda
        #self.Syn = self.netG(self.real_A)
        self.Syn, freqfake = self.netG(self.real_A)     #第一个子网的输出
        # self.un=self.fake_G.clone()
        # self.Unknowregion=self.un.data.masked_fill_(self.inv_ex_mask, 0)
        # self.knownregion=self.real_A.data.masked_fill_(self.ex_mask, 0)
        # self.Syn=self.Unknowregion+self.knownregion

    # def vali(self):
    #     self.netG.eval()
    #     self.real_A = self.input_A.to(self.device)
    #     self.real_B = self.input_B.to(self.device)  # 真实图像利用cuda
    #     self.fake_P= self.netP(self.real_A)     #第一个子网的输出
    #     self.un=self.fake_P.clone()
    #     self.Unknowregion=self.un.data.masked_fill_(self.inv_ex_mask, 0)
    #     self.knownregion=self.real_A.data.masked_fill_(self.ex_mask, 0)
    #     self.Syn=self.Unknowregion+self.knownregion
    #     self.Middle = torch.cat((self.Syn, self.input_A), 1)  # 沿着通道维第一个子网的修复结果和输入残缺图像进行拼接
    #     self.fake_B, self.fake_SCSA1, self.fake_SCSA2, self.fake_SA1, self.fake_SA2 = self.netG(self.Middle)            #第二个子网操作
    #     self.loss_G_L1ceshi = self.criterionL1(self.fake_B, self.real_B)


    def backward_D(self):
        fake_AB = self.Syn      #第二个子网的输出
        # Real

        real_AB = self.real_B # GroundTruth

        self.pred_fake = self.netD(fake_AB.detach())   #修复图像输入到批鉴别器
        self.pred_real = self.netD(real_AB)            #真实图像输入到批鉴别器
        self.loss_D_fake = self.criterionGAN(self.pred_fake, self.pred_real, True)   #计算批鉴别器GAN损失

        # When two losses are ready, together backward.
        # It is op, so the backward will be called from a leaf.(quite different from LuaTorch)
        self.loss_D_fake.backward()


    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = self.Syn

        self.gt_latent_fake = self.vgg(Variable(self.Syn.data, requires_grad=False))
        self.gt_latent_real = self.vgg(Variable(self.input_B, requires_grad=False))

        pred_fake = self.netD(fake_AB)
        pred_real = self.netD(self.real_B)

        self.loss_G_sty = self.criterionL1(self.compute_gram(self.gt_latent_fake.relu2_2), self.compute_gram(self.gt_latent_real.relu2_2)) + self.criterionL1(self.compute_gram(self.gt_latent_fake.relu3_2), self.compute_gram(self.gt_latent_real.relu3_2)) + \
                          self.criterionL1(self.compute_gram(self.gt_latent_fake.relu4_2), self.compute_gram(self.gt_latent_real.relu4_2)) + self.criterionL1(self.compute_gram(self.gt_latent_fake.relu5_2), self.compute_gram(self.gt_latent_real.relu5_2)) + \
                          self.criterionL1(self.compute_gram(self.gt_latent_fake.relu1_2), self.compute_gram(self.gt_latent_real.relu1_2))
        self.loss_G_per = self.criterionL1(self.gt_latent_fake.relu1_1, self.gt_latent_real.relu1_1) + self.criterionL1(self.gt_latent_fake.relu2_1, self.gt_latent_real.relu2_1) + \
                          self.criterionL1(self.gt_latent_fake.relu3_1, self.gt_latent_real.relu3_1) + self.criterionL1(self.gt_latent_fake.relu4_1, self.gt_latent_real.relu4_1) + \
                          self.criterionL1(self.gt_latent_fake.relu5_1, self.gt_latent_real.relu5_1)
        self.loss_G_GAN = self.criterionGAN(pred_fake,pred_real, False)
        self.loss_G_L1 =self.criterionL1(self.Syn, self.real_B)# + self.criterionL1(self.freqfake, self.real_B)   #重建损失self.freqfake
        self.loss_G_Frequency = self.frequencyLoss.loss_formulation(self.freqfake, self.input_f,matrix=None)
        self.loss_G_Frequency = self.loss_G_Frequency.item()
        self.losses_H,self.losses_S,self.losses_V,self.losses = HSV.HSV(self.real_B,self.Syn)
        self.loss_G_HS = self.losses_H * self.opt.weight_losses_H  +  self.losses_S * self.opt.weight_losses_S
        self.loss_G_edge = self.Edge_MSE_loss(self.real_B, self.Syn, self.edgeSyn)
        self.loss_G = self.loss_G_L1 * self.opt.lambda_A + self.loss_G_GAN * self.opt.gan_weight + self.loss_G_per * self.opt.per_weight + self.loss_G_sty * self.opt.style_weight + self.opt.weight_losses_HS * self.losses_G_HS + self.opt.weight_loss_G_edge * self.losses_G_edge#+ self.opt.frequency * self.loss_G_Frequency# * self.opt.frequency_weight
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_D.zero_grad()     #反向传播前清理梯度
        self.backward_D()
        torch.nn.utils.clip_grad_norm_(parameters=self.netG.parameters(), max_norm=10, norm_type=2)
        self.optimizer_D.step()        #模型更新
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


    def get_current_errors(self):
        return OrderedDict([('GANLoss', self.loss_G_GAN.data.item()),
                            ('Recloss', self.loss_G_L1.data.item()),
                            ('Perpetualloss', self.loss_G_per.data.item()),
                            ('Styleloss', self.loss_G_sty.data.item()),
                            ('frequencyloss', self.loss_G_Frequency),
                            ('Dloss', self.loss_D_fake.data.item()),
                            ])


    def lossxianshi(self):
        return self.loss_G_L1ceshi.data.item()


    def jisuanssim(self,yuanshi,duizhao):
        original = cv2.imread(yuanshi)  # numpy.adarray
        contrast = cv2.imread(duizhao)
        image1 = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY) #  将图像转换为灰度图
        image2 = cv2.cvtColor(contrast,cv2.COLOR_BGR2GRAY) #  将图像转换为灰度图
        sim = ssim(image1, image2)
        mse = np.mean( (original/255. - contrast/255.) ** 2 )
        if mse < 1.0e-10:
            psnrValue = 100
        PIXEL_MAX = 1
        psnrValue = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        return sim,psnrValue


    def jisuanLPIPS(self,yuanshi,duizhao):
        using_gpu = True

        # Linearly calibrated models (LPIPS)
        loss_fn = lpips.LPIPS(net='alex', spatial=True)

        if (using_gpu):
            loss_fn.cuda()#to(self.device)

        ## Example usage with images
        ex_ref = lpips.im2tensor(lpips.load_image(yuanshi))
        ex_p0 = lpips.im2tensor(lpips.load_image(duizhao))
        if (using_gpu):
            ex_ref = ex_ref.cuda()#to(self.device)
            ex_p0 = ex_p0.cuda()#to(self.device)

        ex_d0 = loss_fn.forward(ex_ref, ex_p0)
        lpips_jieguo = ex_d0.mean()
        lpips_jieguo = lpips_jieguo.cpu()
        lpips_jieguo = lpips_jieguo.detach().numpy()

        return lpips_jieguo


    def get_current_visuals(self):

        self.un=self.Syn.clone()
        self.Unknowregion=self.un.data.masked_fill_(self.inv_ex_mask, 0)
        self.knownregion=self.real_A.data.masked_fill_(self.ex_mask, 0)
        self.fake_=self.Unknowregion+self.knownregion

        real_A =self.real_A.data   #残缺图像
        fake_ = self.fake_.data  #修复后的图像
        real_B =self.real_B.data   #真实图像

        return real_A,real_B,fake_


    def save(self, epoch):
        self.save_network(self.netG, 'G', epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', epoch, self.gpu_ids)

    def load(self, epoch):
        self.load_network(self.netG, 'G', epoch)
        
