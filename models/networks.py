import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from torch.autograd import Variable
import functools
from models import MHSA
from models import SCA
from models import GFT
import torch.nn.functional as F
from torch.optim import lr_scheduler
import math
import torch.fft as fft
import cv2
from models import moganet
from skimage.color import rgb2gray
from skimage.feature import canny
# from models import swin_transformer_acmix
#from timm.models.layers import DropPath, to_2tuple, trunc_normal_





###############################################################################
# Functions
###############################################################################

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)  # 自定义调整学习率
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):  # 参数初始化
    def init_func(m):
        classname = m.__class__.__name__  # 对网络的每个module进行参数初始化
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda()
    init_weights(net, init_type, gain=init_gain)  # 对网络参数初始化
    return net


def define_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False,
             init_type='normal', gpu_ids=[], init_gain=0.02):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)  # 定义归一化处理函数

    netG = UnetGeneratorWSA(input_nc, output_nc, ngf, norm_layer=norm_layer)

    return init_net(netG, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[], init_gain=0.02):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)  # 选择归一化处理方式

    netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer,
                                   use_sigmoid=use_sigmoid)  # 3  64 3 归一化 False

    return init_net(netD, init_type, init_gain, gpu_ids)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# GANLosses
##############################################################################


class GANLoss(nn.Module):
    def __init__(self, gan_type='wgan_gp', target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if gan_type == 'wgan_gp':
            self.loss = nn.MSELoss()
        elif gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_type == 'vanilla':
            self.loss = nn.BCELoss()
        else:
            raise ValueError("GAN type [%s] not recognized." % gan_type)

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)  # 用1填充
                self.real_label_var = Variable(real_tensor, requires_grad=False)  # 指定该节点及依赖它的节点不需要求导
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, y_pred_fake, y_pred, target_is_real):
        target_tensor = self.get_target_tensor(y_pred_fake, target_is_real)
        if (target_is_real):
            errD = (torch.mean((y_pred - torch.mean(y_pred_fake) - target_tensor) ** 2) + torch.mean(
                (y_pred_fake - torch.mean(y_pred) + target_tensor) ** 2)) / 2
            return errD


        else:
            errG = (torch.mean((y_pred - torch.mean(y_pred_fake) + target_tensor) ** 2) + torch.mean(
                (y_pred_fake - torch.mean(y_pred) - target_tensor) ** 2)) / 2
            return errG

##############################################################################
# Generator
##############################################################################

# class unetDown(nn.Module):
#     def __init__(self, in_size, out_size, norm_layer=nn.BatchNorm2d):
#         super(unetDown, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1),
#             #gnconv(out_size),
#             norm_layer(out_size, affine=True),
#             nn.LeakyReLU(0.2, True),)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(out_size, out_size, kernel_size=4, stride=2, padding=1),
#             #gnconv(out_size),
#             norm_layer(out_size, affine=True),
#             nn.ELU(),)

#     def forward(self, inputs):
#         outputs = self.conv1(inputs)
#         outputs = self.conv2(outputs)
#         return outputs

class unetDown(nn.Module):
    def __init__(self, in_size, out_size, norm_layer=nn.BatchNorm2d):
        super(unetDown, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1),
            norm_layer(out_size, affine=True),
            nn.LeakyReLU(0.2, True),
            gnconv(out_size),
            norm_layer(out_size, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_size, out_size, kernel_size=4, stride=2, padding=1),
            norm_layer(out_size, affine=True),
            nn.LeakyReLU(0.2, True)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=1, stride=2),
            norm_layer(out_size, affine=True),
            nn.LeakyReLU(0.2, True),
            gnconv(out_size),
            norm_layer(out_size, affine=True),
            nn.LeakyReLU(0.2, True),
        )
        self.relu = nn.ReLU(inplace=True)
        
        self.mogalayer = moganet.MogaBlock(out_size,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_type='GELU',
                 norm_type='BN',
                 init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False)
        
    def forward(self, inputs):
        residual = inputs
        outputs = self.conv1(inputs)
        outputs = self.mogalayer(outputs)
        outputs = outputs + self.shortcut(residual)
        #outputs = self.relu(outputs)
        return outputs

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, norm_layer=nn.BatchNorm2d, isLast=False):
        super(unetUp, self).__init__()
        self.conv1 = nn.Sequential(
                nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=1, padding=1),
                norm_layer(out_size, affine=True),
                nn.ReLU(True),
                gnconv(out_size),
                norm_layer(out_size, affine=True),
                nn.LeakyReLU(0.2, True))
        if isLast:
            #self.conv2 = nn.ConvTranspose2d(out_size, out_size, kernel_size=4, stride=2, padding=1)
            self.conv2 = nn.Sequential(nn.ConvTranspose2d(out_size, out_size, kernel_size=4, stride=2, padding=1),
                     norm_layer(out_size, affine=True),
                     nn.ReLU(True),
                     gnconv(out_size),
                     norm_layer(out_size, affine=True),
                     nn.LeakyReLU(0.2, True))
        else:
            self.conv2 = nn.Sequential(
                nn.ConvTranspose2d(out_size, out_size, kernel_size=4, stride=2, padding=1),
                norm_layer(out_size, affine=True),
                nn.ReLU(True),
                gnconv(out_size),
                norm_layer(out_size, affine=True),
                nn.LeakyReLU(0.2, True))
        self.mogalayer = moganet.MogaBlock(out_size,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_type='GELU',
                 norm_type='BN',
                 init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False)

    def forward(self, inputs1, inputs2):
        _, _, h, w = inputs1.size()
        if h != inputs2.size(2) or w != inputs2.size(3):
            inputs2 = F.upsample(inputs2, (h, w), mode='bilinear')
        finalout = torch.cat([inputs2, inputs1], 1)  # cat in the C channel
        finalout = self.conv1(finalout)
        finalout = self.mogalayer(finalout)
        finalout = self.conv2(finalout)
        return finalout

class unetUp_end(nn.Module):
    def __init__(self, in_size, out_size, norm_layer=nn.BatchNorm2d, isLast=False):
        super(unetUp_end, self).__init__()
        self.conv1 = nn.Sequential(
                nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=1, padding=1),
                norm_layer(out_size, affine=True),
                nn.ReLU(True))
        if isLast:
            #self.conv2 = nn.ConvTranspose2d(out_size, out_size, kernel_size=4, stride=2, padding=1)
            self.conv2 = nn.Sequential(nn.ConvTranspose2d(out_size, out_size, kernel_size=4, stride=2, padding=1),
                     norm_layer(out_size, affine=True),
                     nn.ReLU(True))
        else:
            self.conv2 = nn.Sequential(
                nn.ConvTranspose2d(out_size, out_size, kernel_size=4, stride=2, padding=1),
                norm_layer(out_size, affine=True),
                nn.ReLU(True))
        self.mogalayer = moganet.MogaBlock(out_size,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_type='GELU',
                 norm_type='BN',
                 init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[3/8, 9/8, 12/8],
                 attn_act_type='SiLU',
                 attn_force_fp32=False)

    def forward(self, inputs1, inputs2):
        _, _, h, w = inputs1.size()
        if h != inputs2.size(2) or w != inputs2.size(3):
            inputs2 = F.upsample(inputs2, (h, w), mode='bilinear')
        finalout = torch.cat([inputs2, inputs1], 1)  # cat in the C channel
        finalout = self.conv1(finalout)
        #finalout = self.mogalayer(finalout)
        finalout = self.conv2(finalout)
        return finalout

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gnconv = gnconv(out_channels * 2)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfftn(x,s=(h,w),dim=(2,3),norm='ortho')
        ffted = torch.cat([ffted.real,ffted.imag],dim=1)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        
        ffted = self.gnconv(ffted)
        
        ffted = self.relu(self.bn(ffted))

        ffted = torch.tensor_split(ffted,2,dim=1)
        ffted = torch.complex(ffted[0],ffted[1])
        output = torch.fft.irfftn(ffted,s=(h,w),dim=(2,3),norm='ortho')

        output = self.gamma * output + x

        return output

def get_dwconv(dim, kernel, bias):
    return nn.Sequential(nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim),
                        torch.nn.BatchNorm2d(dim),
                        nn.ELU(inplace=True))

class gnconv(nn.Module):
     def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
         super().__init__()
         self.order = order
         self.dims = [dim // 2 ** i for i in range(order)]
         self.dims.reverse()
         self.proj_in = nn.Sequential(nn.Conv2d(dim, 2*dim, 1),torch.nn.BatchNorm2d(2*dim),
                                    nn.ELU(inplace=True))
 
         if gflayer is None:
             self.dwconv = get_dwconv(sum(self.dims), 7, True)
         else:
             self.dwconv = gflayer(sum(self.dims), h=h, w=w) 
         self.proj_out = nn.Sequential(nn.Conv2d(dim, dim, 1),
                                    torch.nn.BatchNorm2d(dim),
                                    nn.ELU(inplace=True)) 
         self.pws = nn.ModuleList(
             [nn.Conv2d(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
         ) 
         self.scale = s
         print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f'%self.scale)
 
 
     def forward(self, x, mask=None, dummy=False):
         B, C, H, W = x.shape 
         fused_x = self.proj_in(x)
         #print("!!!!!!!!!!!!!!",fused_x)
         pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
         dw_abc = self.dwconv(abc) * self.scale 
         #print("??????????????",dw_abc)
         dw_list = torch.split(dw_abc, self.dims, dim=1)
         #print("...............",dw_list)
         x = pwa * dw_list[0] 
         for i in range(self.order -1):
             x = self.pws[i](x) * dw_list[i+1] 
         x = self.proj_out(x) 
        #  print(x)
        #  print(x.shape)
         return x
     
class gnconv_end(nn.Module):
     def __init__(self, dim, order=1, gflayer=None, h=14, w=8, s=1.0):
         super().__init__()
         self.order = order
         self.dims = [dim // 2 ** i for i in range(order)]
         self.dims.reverse()
         self.proj_in = nn.Sequential(nn.Conv2d(dim, 2*dim, 1),torch.nn.BatchNorm2d(2*dim),
                                    nn.ELU(inplace=True))
 
         if gflayer is None:
             self.dwconv = get_dwconv(sum(self.dims), 7, True)
         else:
             self.dwconv = gflayer(sum(self.dims), h=h, w=w) 
         self.proj_out = nn.Sequential(nn.Conv2d(dim, dim, 1),
                                    torch.nn.BatchNorm2d(dim),
                                    nn.ELU(inplace=True)) 
         self.pws = nn.ModuleList(
             [nn.Conv2d(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
         ) 
         self.scale = s
         print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f'%self.scale)
 
 
     def forward(self, x, mask=None, dummy=False):
         B, C, H, W = x.shape 
         fused_x = self.proj_in(x)
         #print("!!!!!!!!!!!!!!",fused_x)
         pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
         dw_abc = self.dwconv(abc) * self.scale 
         #print("??????????????",dw_abc)
         dw_list = torch.split(dw_abc, self.dims, dim=1)
         #print("...............",dw_list)
         x = pwa * dw_list[0] 
         for i in range(self.order -1):
             x = self.pws[i](x) * dw_list[i+1] 
         x = self.proj_out(x) 
        #  print(x)
        #  print(x.shape)
         return x

def fft_compute_color(img_col, center=True):
    assert img_col.size(1) == 3, "Should be color image"
    #print(img_col.size())
    
    batch_size, _, h, w = img_col.size()
    lims_list = []
    idx_list_ = []
    x_mag = torch.zeros((batch_size, 3, h, w))
    x_phase = torch.zeros((batch_size, 3, h, w))
    x_fft = torch.zeros((batch_size, 6, h, w))
    
    for i in range(batch_size):
        for j in range(3):
            img = img_col[i, j]
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
    
    return x_fft, x_mag, x_phase, idx_list_

def fft_compute_conv3(img_col, center=True):
    assert img_col.size(1) == 128, " number of channels Should be same as conv3"
    
    batch_size, _, h, w = img_col.size()
    lims_list = []
    idx_list_ = []
    x_mag = torch.zeros((batch_size, 128, h, w))  # Change the channel dimension to 128 for x_mag
    x_phase = torch.zeros((batch_size, 128, h, w))  # Change the channel dimension to 128 for x_phase
    x_fft = torch.zeros((batch_size, 128, h, w))  # Change the channel dimension to 128 for x_fft
    
    for i in range(batch_size):
        for j in range(3):
            img = img_col[i, j]
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
        
            x_fft[i, j] = dft.real
            x_fft[i, j+3] = dft.imag
        
            idx_list_.append(idx)
    
    return x_fft, x_mag, x_phase

def fft_compute_iconv3(img_col, center=True):
    assert img_col.size(1) == 64, " number of channels Should be same as iconv3"
    
    batch_size, _, h, w = img_col.size()
    lims_list = []
    idx_list_ = []
    x_mag = torch.zeros((batch_size, 64, h, w))  # Change the channel dimension to 128 for x_mag
    x_phase = torch.zeros((batch_size, 64, h, w))  # Change the channel dimension to 128 for x_phase
    x_fft = torch.zeros((batch_size, 64, h, w))  # Change the channel dimension to 128 for x_fft
    
    for i in range(batch_size):
        for j in range(3):
            img = img_col[i, j]
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
        
            x_fft[i, j] = dft.real
            x_fft[i, j+3] = dft.imag
        
            idx_list_.append(idx)
    
    return x_fft, x_mag, x_phase
def ifft_compute_color(x_mag, x_phase, center=True):
    batch_size, _, h, w = x_mag.size()
    recon_im = torch.zeros((batch_size, 3, h, w))

    for i in range(batch_size):
        for j in range(3):
            magnitude = torch.exp(x_mag[i, j])
            phase = x_phase[i, j]

            complex_signal = magnitude * torch.exp(1j * phase)

            ifft = torch.fft.ifft2(complex_signal, dim=(-2, -1))

            if center:
                ifft_unshifted = torch.fft.ifftshift(ifft, dim=(-2, -1))
            else:
                ifft_unshifted = ifft

            img_reconstructed = ifft_unshifted.real

            recon_im[i, j] = img_reconstructed

    return recon_im

def ifft_compute_mag_down(x_mag, center=True):
    batch_size, _, h, w = x_mag.size()
    recon_im = torch.zeros((batch_size, 128, h, w))

    for i in range(batch_size):
        for j in range(3):
            magnitude = torch.exp(x_mag[i, j])
            phase = torch.zeros_like(magnitude)

            complex_signal = magnitude * torch.exp(1j * phase)

            ifft = torch.fft.ifft2(complex_signal, dim=(-2, -1))

            if center:
                ifft_unshifted = torch.fft.ifftshift(ifft, dim=(-2, -1))
            else:
                ifft_unshifted = ifft

            img_reconstructed = ifft_unshifted.real

            recon_im[i, j] = img_reconstructed

    return recon_im

def ifft_compute_phase_down(x_phase, center=True):
    batch_size, _, h, w = x_phase.size()
    recon_im = torch.zeros((batch_size, 128, h, w))

    for i in range(batch_size):
        for j in range(3):
            phase = x_phase[i, j]

            complex_signal = torch.exp(1j * phase)

            ifft = torch.fft.ifft2(complex_signal, dim=(-2, -1))

            if center:
                ifft_unshifted = torch.fft.ifftshift(ifft, dim=(-2, -1))
            else:
                ifft_unshifted = ifft

            img_reconstructed = ifft_unshifted.real

            recon_im[i, j] = img_reconstructed

    return recon_im

def ifft_compute_mag_up(x_mag, center=True):
    batch_size, _, h, w = x_mag.size()
    recon_im = torch.zeros((batch_size, 64, h, w))

    for i in range(batch_size):
        for j in range(3):
            magnitude = torch.exp(x_mag[i, j])
            phase = torch.zeros_like(magnitude)

            complex_signal = magnitude * torch.exp(1j * phase)

            ifft = torch.fft.ifft2(complex_signal, dim=(-2, -1))

            if center:
                ifft_unshifted = torch.fft.ifftshift(ifft, dim=(-2, -1))
            else:
                ifft_unshifted = ifft

            img_reconstructed = ifft_unshifted.real

            recon_im[i, j] = img_reconstructed

    return recon_im

def ifft_compute_phase_up(x_phase, center=True):
    batch_size, _, h, w = x_phase.size()
    recon_im = torch.zeros((batch_size, 64, h, w))

    for i in range(batch_size):
        for j in range(3):
            magnitude = torch.exp(x_phase[i, j])
            phase = torch.zeros_like(magnitude)

            complex_signal = magnitude * torch.exp(1j * phase)

            ifft = torch.fft.ifft2(complex_signal, dim=(-2, -1))

            if center:
                ifft_unshifted = torch.fft.ifftshift(ifft, dim=(-2, -1))
            else:
                ifft_unshifted = ifft

            img_reconstructed = ifft_unshifted.real

            recon_im[i, j] = img_reconstructed

    return recon_im

class ResNetDownBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ELU(inplace = True),gnconv(out_channels),
                        nn.BatchNorm2d(out_channels),
                        nn.ELU(inplace = True)
                        )
            self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
           ,nn.BatchNorm2d(out_channels),nn.ELU(inplace = True),gnconv(out_channels),
                        nn.BatchNorm2d(out_channels),
                        nn.ELU(inplace = True))
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2,padding=1,bias=False),
                nn.BatchNorm2d(out_channels),nn.ELU(inplace = True),gnconv(out_channels),
                        nn.BatchNorm2d(out_channels),
                        nn.ELU(inplace = True) )
                                                           

            self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),  # 16表示r，filter3//16表示C/r，这里用卷积层代替全连接层
            nn.ReLU(),
            nn.Conv2d(out_channels // 16,out_channels, kernel_size=1),
            nn.Sigmoid()
        )    
            self.mogalayer = moganet.MogaBlock( out_channels,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_type='GELU',
                 norm_type='BN',
                 init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False)                                           

        def forward(self, x):
            out = self.conv1(x)
            #out = self.Vit(x)
            out = self.conv2(out)
            out = self.mogalayer(out)
            # weights = self.se(out)
            # out = out * weights
            # print("----------------------resnet输出x------------------------")
            # print(x)
            # print("----------------------resnet输出out------------------------")
            # print(out)
            # print("----------------------resnet输出------------------------")
            out = out+self.shortcut(x)
            #print("----------------------resnet输出relu------------------------")
            
            return out


class ResNetUPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ELU(inplace=True),
                    gnconv(out_channels),
                    nn.BatchNorm2d(out_channels),
                    nn.ELU(inplace = True))
        #self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ELU(inplace=True),
                    gnconv(out_channels),
                    nn.BatchNorm2d(out_channels),
                    nn.ELU(inplace = True))
        #self.bn2 = nn.BatchNorm2d(out_channels)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
                gnconv(out_channels),
                nn.BatchNorm2d(out_channels),
                nn.ELU(inplace = True)
            )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            gnconv(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )
        self.mogalayer = moganet.MogaBlock( out_channels,
                 ffn_ratio=3,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_type='GELU',
                 norm_type='BN',
                 init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False)                                           


    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv1(x)
        out = self.conv2(out)
        #out = F.relu(self.bn1(gnconv(out)))
        #out = self.bn2(self.conv2(out))
        #out = F.relu(self.bn1(gnconv(out)))
        out = self.mogalayer(out)
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.upsample(out)
        return out

class ResNetUPBlock_end(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ELU(inplace=True)
                )
        #self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ELU(inplace=True)
                    )
        #self.bn2 = nn.BatchNorm2d(out_channels)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),nn.ELU(inplace=True)
                            )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            
        )
        self.mogalayer = moganet.MogaBlock( out_channels,
                 ffn_ratio=3,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_type='GELU',
                 norm_type='BN',
                 init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[3/8, 9/8, 12/8],
                 attn_act_type='SiLU',
                 attn_force_fp32=False)                                           


    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out = self.conv1(x)
        out = self.conv2(out)
       # out = self.mogalayer(out)
        out =out +  self.shortcut(x)
        #out = F.relu(out)
        out = self.upsample(out)
        return out

class SFAttention(nn.Module):
    def __init__(self, in_channels):
        super(SFAttention, self).__init__()
        
        self.query_conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1),
                                        nn.BatchNorm2d(in_channels),
                                        nn.ELU(inplace = True))
        self.key_conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1),
                                        nn.BatchNorm2d(in_channels),
                                        nn.ELU(inplace = True))
        self.value_conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1),
                                        nn.BatchNorm2d(in_channels),
                                        nn.ELU(inplace = True))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1,bias=False),
                nn.BatchNorm2d(in_channels),nn.ELU(inplace = True) )
        
    def forward(self, x_q, x_k, x_v):
        batch_size, channels, height_q, width_q = x_q.size()
        _, _, height_k, width_k = x_k.size()
        _, _, height_v, width_v = x_v.size()
        
        query = self.query_conv(x_q).view(batch_size, -1, height_q * width_q).permute(0, 2, 1)
        key = self.key_conv(x_k).view(batch_size, -1, height_k * width_k)
        value = self.value_conv(x_v).view(batch_size, -1, height_v * width_v)
        
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height_v, width_v)
        out = self.gamma * out + self.beta*x_v#self.beta * self.shortcut(x_v)
        return out




# class FastFourierConv(nn.Module):
#     def __init__(self, input_size, out_size, norm_layer=nn.BatchNorm2d, num_heads=2, isDown=True):
#         super(FastFourierConv, self).__init__()

#         if isDown:
#             self.conv1 = nn.Sequential(
#                 nn.Conv2d(input_size, input_size, kernel_size=4, stride=2, padding=1),
#                 #gnconv(input_size),
#                 norm_layer(input_size, affine=True),
#                 nn.LeakyReLU(0.2, True), )
#         else:
#             self.conv1 = nn.Sequential(
#                 nn.ConvTranspose2d(input_size, input_size, kernel_size=4, stride=2, padding=1),
#                 #gnconv(input_size),
#                 norm_layer(input_size, affine=True),
#                 nn.ReLU(True), )

#         self.ffc1 = FourierUnit(input_size, input_size, groups=1)
#         self.mltSA = MHSA.Block(dim=input_size, num_heads=num_heads)

#         if isDown:
#             self.conv2 = nn.Sequential(
#                 nn.Conv2d(input_size, out_size, kernel_size=4, stride=2, padding=1),
#                 #gnconv(out_size),
#                 norm_layer(out_size, affine=True),
#                 nn.LeakyReLU(0.2, True), )
#         else:
#             self.conv2 = nn.Sequential(
#                 nn.ConvTranspose2d(input_size, out_size, kernel_size=4, stride=2, padding=1),
#                 #gnconv(out_size),
#                 norm_layer(out_size, affine=True),
#                 nn.ReLU(True), )

#         self.ffc2 = FourierUnit(out_size, out_size, groups=1)
#         self.conv3 = nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=1)

#     def forward(self, x):
#         conv1 = self.conv1(x)
#         ffc1 = self.ffc1(conv1)
#         mltSA = self.mltSA(ffc1)
#         conv2 = self.conv2(mltSA)
#         ffc2 = self.ffc2(conv2)
#         conv3 = self.conv3(ffc2)
#         return conv3

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
        dilation=1, groups=1, relu=True, bn=True,bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, momentum=0.01) 

        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
            return x

class ZPool(nn.Module):
    def forward(self, x):
        y = torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        # print('zhelizhelizhelizhelizhelizhelizhelizhelizheli')
        # print(y.shape)
        # print('zhelizhelizhelizhelizhelizhelizhelizhelizheli')
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=True)
    def forward(self, x):
        x_compress = self.compress(x)
        # print('222222222222222222222222222222')
        # if x_compress is not None:
        #     print(x_compress.shape)
        # print('222222222222222222222222222222')
        x_out = self.conv(x_compress)

        scale = torch.sigmoid_(x_out) 
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out

        
    
class UnetGeneratorWSA(nn.Module):#主网络结构
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        super(UnetGeneratorWSA, self).__init__()

        # construct unet structure
        self.conv1 = unetDown(input_nc, ngf, norm_layer=norm_layer)
        self.conv2 = unetDown(ngf, ngf, norm_layer=norm_layer)
        self.conv3 = unetDown(ngf, ngf * 2, norm_layer=norm_layer)
        self.conv4 = unetDown(ngf * 2, ngf * 2, norm_layer=norm_layer)
        self.conv5 = unetDown(ngf * 2, ngf * 4, norm_layer=norm_layer)
        self.conv6 = unetDown(ngf * 4, ngf * 4, norm_layer=norm_layer)
        self.conv7 = unetDown(ngf * 4, ngf * 8, norm_layer=norm_layer)
        self.center1 = nn.Sequential(
                nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, True),gnconv(ngf*8),nn.BatchNorm2d(ngf*8),
                                        nn.ELU(inplace = True))
        self.center2 = nn.Sequential(
                nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
                norm_layer(ngf * 8, affine=True),
                nn.LeakyReLU(0.2, True) )
        self.iconv7 = unetUp(ngf * 16, ngf * 4, norm_layer=norm_layer, isLast=False)
        self.iconv6 = unetUp(ngf * 8, ngf * 4, norm_layer=norm_layer, isLast=False)
        self.iconv5 = unetUp(ngf * 8, ngf * 2, norm_layer=norm_layer, isLast=False)
        self.iconv4 = unetUp(ngf * 4, ngf * 2, norm_layer=norm_layer, isLast=False)
        self.iconv3 = unetUp(ngf * 4, ngf, norm_layer=norm_layer, isLast=False)
        self.iconv2 = unetUp(ngf * 2, ngf, norm_layer=norm_layer, isLast=False)
        self.iconv1 = unetUp_end(ngf * 2, output_nc, norm_layer=norm_layer, isLast=True)
        # construct frequency_transformer structure
        self.fdownconv1 = ResNetDownBlock(input_nc, ngf)
        self.fdownconv2 =  ResNetDownBlock(ngf, ngf*2)
        self.fupconv2 = ResNetUPBlock(ngf*2, ngf)
        self.fupconv1 = ResNetUPBlock_end(ngf, output_nc)
        self.upsample = nn.Sequential(
                nn.ConvTranspose2d(output_nc, output_nc, kernel_size=2, stride=2, padding=0),
                #gnconv(out_size),
                nn.BatchNorm2d(output_nc),
                nn.LeakyReLU(0.2,True) )
        # self.fconv1 = FourierUnit(ngf*4, ngf*4, groups=1)
        # self.fconv2 = FourierUnit(ngf*4, ngf*4, groups=1)
        # self.fconv3 = FourierUnit(ngf*4, ngf*4, groups=1)
        # self.fconv4 = FourierUnit(ngf*4, ngf*4, groups=1)
        #self.mltSA = MHSA.Block(dim=ngf*4, num_heads=2)
        self.sf_attention_middle = SFAttention(in_channels=ngf*2)
        self.sf_attention_middle2 = SFAttention(in_channels=ngf)
        #self.sf_attention_spatial = SFAttention(in_channels=ngf * 8)
        self.sf_attention_end = SFAttention(in_channels=output_nc)
        
        self.attention_frequency_middle = TripletAttention(no_spatial=False)
        
        #self.attention_frequency_end = swin_transformer_acmix.WindowAttention_acmix(dim=3, window_size=to_2tuple(self.window_size), num_heads=2, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.)
        #self.SCA1 = SCA.SCA(output_nc)
        # self.SCA1 = SCA.SCA(ngf)
        # self.fftran1 = FastFourierConv(ngf, ngf * 2, norm_layer=norm_layer, num_heads=2, isDown=True)
        # self.SCA2 = SCA.SCA(ngf * 2)
        # self.fftran2 = FastFourierConv(ngf * 2, ngf * 4, norm_layer=norm_layer, num_heads=4, isDown=True)
        # self.SCA3 = SCA.SCA(ngf * 4)
        # self.fftran3 = FastFourierConv(ngf * 4, ngf * 8, norm_layer=norm_layer, num_heads=8, isDown=True)
        # self.tranCenter1 = nn.Sequential(
        #         nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
        #         nn.LeakyReLU(0.2, True),)
        # self.tranCenter2 = nn.Sequential(
        #         nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
        #         norm_layer(ngf * 8, affine=True),
        #         nn.ReLU(True), )
        # self.fftrani3 = FastFourierConv(ngf * 8, ngf * 4, norm_layer=norm_layer, num_heads=8, isDown=False)
        # self.GFT3 = GFT.GFT(ngf * 4)
        # self.fftrani2 = FastFourierConv(ngf * 4, ngf * 2, norm_layer=norm_layer, num_heads=4, isDown=False)
        # self.GFT2 = GFT.GFT(ngf * 2)
        # self.fftrani1 = FastFourierConv(ngf * 2, ngf, norm_layer=norm_layer, num_heads=2, isDown=False)
        self.GFT1 = GFT.GFT(ngf*2)
        self.SCA1 = SCA.SCA(ngf*2)
        self.SCA2 = SCA.SCA(ngf)
        self.GFT2 = GFT.GFT(ngf)
        self.GFT3 = GFT.GFT(output_nc)
        self.dropout = nn.Dropout(0.3)
        # self.transIConv1 = nn.ConvTranspose2d(ngf, output_nc, kernel_size=4, stride=2, padding=1)
        
    def load_edge(self, img):
        return canny(img, sigma=2, mask=None).astype(np.float)


    def forward(self, input):
        conv1 = self.conv1(input)    
        conv2 = self.conv2(conv1) 
        #conv2 = self.dropout(conv2)   
        conv3 = self.conv3(conv2)   
        conv3_ffted, conv3_ffted_mag, conv3_ffted_phase = fft_compute_conv3(conv3,center=True)
        conv3_ffted = conv3_ffted.cuda()
        conv3_ffted_mag = conv3_ffted_mag.cuda()
        conv3_ffted_phase = conv3_ffted_phase.cuda()      
        #频域支路
        input_fft, input_mag, input_phase, idx_list = fft_compute_color(input, center=True)
        #conv2_ffted, conv2_ffted_mag, conv2_ffted_phase = fft_compute_conv2(conv2,center=True)
        input_fft = input_fft.cuda()                        
        input_mag = input_mag.cuda()
        input_phase = input_phase.cuda() 
        # print('---------------------------------------------')
        # print('input_mag是这样的',input_mag)
        # print('---------------------------------------------')              
        fdownconv1_mag = self.fdownconv1(input_mag)#128*128             
        fdownconv2_mag = self.fdownconv2(fdownconv1_mag)#64*64
        fdownconv2_mag = self.conv4(fdownconv2_mag)#通道数不变，H,W变为一半#32*32       
        SA_mag = self.sf_attention_middle(conv3_ffted_mag,conv3_ffted_mag,fdownconv2_mag)
        SA_mag = self.dropout(SA_mag) 
            
        fdownconv1_phase = self.fdownconv1(input_phase)
        fdownconv2_phase = self.fdownconv2(fdownconv1_phase)
        fdownconv2_phase = self.conv4(fdownconv2_phase)#通道数不变，H,W变为一半   
        SA_phase = self.sf_attention_middle(conv3_ffted_phase,conv3_ffted_phase,fdownconv2_phase)
        SA_phase = self.dropout(SA_phase)
        ######################频域到空余的注意力，下采样时##################################
        fdownconv2_mag_iffted = ifft_compute_mag_down(fdownconv2_mag,center=True)
        fdownconv2_phase_iffted = ifft_compute_phase_down(fdownconv2_phase,center=True)
        fdownconv2_mag_iffted = fdownconv2_mag_iffted.cuda()
        fdownconv2_phase_iffted = fdownconv2_phase_iffted.cuda()
        
        frequencytospace_attention_mag1 = self.sf_attention_middle(fdownconv2_mag_iffted,fdownconv2_mag_iffted, conv3)
        frequencytospace_attention_phase1 = self.sf_attention_middle(fdownconv2_phase_iffted,fdownconv2_phase_iffted, conv3)
        
        conv3 = self.SCA1(frequencytospace_attention_mag1,frequencytospace_attention_phase1)
        #################################################################################
        conv4 = self.conv4(conv3)
        conv4 = self.dropout(conv4) 
        conv5 = self.conv5(conv4)       
        conv6 = self.conv6(conv5)
        conv6 = self.dropout(conv6) 
        conv7 = self.conv7(conv6)
        center1 = self.center1(conv7)
        center1 = self.dropout(center1)              
        center2 = self.center2(center1)
        #center2 = self.sf_attention_spatial(center2,center2,center2)       
        iconv7 = self.iconv7(conv7, center2)
        iconv6 = self.iconv6(conv6, iconv7)        
        iconv5 = self.iconv5(conv5, iconv6)
        iconv5 = self.dropout(iconv5) 
        iconv4 = self.iconv4(conv4, iconv5)
        iconv3 = self.iconv3(conv3, iconv4)
        #iconv3 = self.dropout(iconv3)
        
        #####################对iconv3进行fft变换#####################
        iconv3_ffted, iconv3_ffted_mag, iconv3_ffted_phase = fft_compute_iconv3(iconv3, center=True)
        iconv3_ffted = iconv3_ffted.cuda()
        iconv3_ffted_mag = iconv3_ffted_mag.cuda()
        iconv3_ffted_phase = iconv3_ffted_phase.cuda()
        ############################################################
        
        #SA_phase = self.attention_frequency_middle(fdownconv2_phase)
        #####################幅度支路#####################
        fupconv2_mag = self.fupconv2(SA_mag)
        SA_mag2 = self.sf_attention_middle2(iconv3_ffted_mag,iconv3_ffted_mag,fupconv2_mag)
        fupconv1_mag = self.fupconv1(SA_mag2)
        fupconv1_mag = self.upsample(fupconv1_mag)
        #####################相位支路#####################
        fupconv2_phase = self.fupconv2(SA_phase)
        SA_phase2 = self.sf_attention_middle2(iconv3_ffted_phase,iconv3_ffted_phase,fupconv2_phase)
        fupconv1_phase = self.fupconv1(SA_phase2)        
        fupconv1_phase = self.upsample(fupconv1_phase)
       
        #feature fusion of frequency to space(attention)
        fupconv1_mag_channels = torch.chunk(fupconv1_mag, 3, dim=1)
        fupconv1_phase_channels = torch.chunk(fupconv1_phase, 3, dim=1)
        combined_channels = [torch.cat(( fupconv1_mag_channels[i], fupconv1_phase_channels[i]), dim=1) for i in range(3)]
        freqout = torch.cat(combined_channels, dim=1)
        #freqout = torch.cat((fupconv1_phase, fupconv1_mag), dim=1)    
        # print('频域支路输出-------------------')
        # print(freqout.shape)
        # print('频域支路输出-------------------')
        freqout = freqout.cuda()#频域的输出。没有经过ifft，做频域的损失用

        freqout_iffted = ifft_compute_color(fupconv1_mag, fupconv1_phase)
        #iconv1_ffted, iconv1_ffted_mag, iconv1_ffted_phase,idx_list_notused = fft_compute_color(iconv1,center=True)
        

        freqout_iffted = freqout_iffted.cuda()
        # freqout_iffted_mag = freqout_iffted_mag.cuda()
        # freqout_iffted_phase = freqout_iffted_phase.cuda()
        # iconv1_ffted = iconv1_ffted.cuda()
        # iconv1_ffted_mag = iconv1_ffted_mag.cuda()
        # iconv1_ffted_phase = iconv1_ffted_phase.cuda()

        # frequencytospace_attention_mag1 = self.sf_attention_middle(fdownconv1_mag,fdownconv1_mag, conv3)
        # frequencytospace_attention2 = self.sf_attention_end(freqout_iffted_phase,freqout_iffted_phase, iconv1)
        # spacetofrequency_attention1 = self.sf_attention_end(iconv1_ffted_mag,iconv1_ffted_mag, fupconv1_mag)
        # spacetofrequency_attention2 =self.sf_attention_end(iconv1_ffted_phase,iconv1_ffted_phase, fupconv1_phase)
        # fusion1 = self.GFT1(frequencytospace_attention1,frequencytospace_attention2)
        # fusion2 = self.GFT1(spacetofrequency_attention1,spacetofrequency_attention2)
        # fused_output = self.GFT1(fusion1,fusion2)
        
        fupconv2_mag_iffted = ifft_compute_mag_up(fupconv2_mag,center=True)
        fupconv2_phase_iffted = ifft_compute_phase_up(fupconv2_phase,center=True)
        fupconv2_mag_iffted = fupconv2_mag_iffted.cuda()
        fupconv2_phase_iffted = fupconv2_phase_iffted.cuda()
        
        frequencytospace_attention_mag2 = self.sf_attention_middle2(fupconv2_mag_iffted,fupconv2_mag_iffted, iconv3)
        frequencytospace_attention_phase2 = self.sf_attention_middle2(fupconv2_phase_iffted,fupconv2_phase_iffted, iconv3)
        
        iconv3 = self.SCA2(frequencytospace_attention_mag2,frequencytospace_attention_phase2)       
       
        iconv2 = self.iconv2(conv2, iconv3)       
        iconv1 = self.iconv1(conv1, iconv2)
        fused_output = self.GFT3(iconv1, freqout_iffted)
        #print(freqout_iffted.shape)
        #fused_output = self.sf_attention_end(fupconv1_mag,fupconv1_phase, iconv1)
        
        fdownconv2_phase_iffted_gray = rgb2gray(fdownconv2_phase_iffted)    #讀成灰階 #这句报错说需要先移动到cpu上

        edge_syn = self.load_edge(fdownconv2_phase_iffted_gray)
        return fused_output, freqout, edge_syn
        

##############################################################################
# Discriminator
##############################################################################

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False):  # 3 64 3 选定的归一化方式 False
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # 1:3
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # 2 4
            sequence += [
                nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                # 通道数：输入64、128 输出128、256    32*32  4*4的卷积核  尺寸减半
                # norm_layer(ndf * nf_mult),  # 归一化
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            # norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
