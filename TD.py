#coding= utf_8
from __future__ import print_function
# from scipy.misc import imread, imresize
import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
import time

import torch.utils.data

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_msssim
class TD(nn.Module):
    def __init__(self, device, alpha_sal=0.7):
        super(TD, self).__init__()

        self.alpha_sal = 1 # alpha_sal

        self.laplacian_kernel = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]], dtype=torch.float,
                                             requires_grad=False)
        # [-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]
        # [0., -1., 0.], [-1.,4., -1.], [0., -1., 0.]
        # self.laplacian_kernel = torch.tensor([[-1.,-2.,-1.], [0.,0.,0.,], [1.,2.,1.]], dtype=torch.float,
        #                                      requires_grad=False)

        self.laplacian_kernel = self.laplacian_kernel.view((1,1,3,3))  # Shape format of weight for convolution
        self.laplacian_kernel = self.laplacian_kernel.to(device)

    @staticmethod
    def LaplaceAlogrithm(image, cuda_visible=True):
        assert torch.is_tensor(image) is True
        #[-1.,-2.,-1.], [0.,0.,0.,], [1.,2.,1.]
        #[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]
        #[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1]
        laplace_operator = np.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=np.float32)[np.newaxis, :, :].repeat(1, 0)
        if cuda_visible:
            laplace_operator = torch.from_numpy(laplace_operator).unsqueeze(0).cuda()
        else:
            laplace_operator = torch.from_numpy(laplace_operator).unsqueeze(0)

        image =  F.conv2d(image, laplace_operator, padding=1, stride=1)
        return image
    @staticmethod
    def SobelOperator(image, cuda_visible=True):
        assert torch.is_tensor(image) is True

        # 定义Sobel算子的水平和垂直核
        sobel_operator_horizontal = np.array([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ], dtype=np.float32)[np.newaxis, :, :].repeat(1, 0)

        sobel_operator_vertical = np.array([
            [-1., -2., -1.],
            [0., 0., 0.],
            [1., 2., 1.]
        ], dtype=np.float32)[np.newaxis, :, :].repeat(1, 0)

        if cuda_visible:
            sobel_operator_horizontal = torch.from_numpy(sobel_operator_horizontal).unsqueeze(0).cuda()
            sobel_operator_vertical = torch.from_numpy(sobel_operator_vertical).unsqueeze(0).cuda()
        else:
            sobel_operator_horizontal = torch.from_numpy(sobel_operator_horizontal).unsqueeze(0)
            sobel_operator_vertical = torch.from_numpy(sobel_operator_vertical).unsqueeze(0)

        # 应用Sobel算子
        image_horizontal = F.conv2d(image, sobel_operator_horizontal, padding=1, stride=1)
        image_vertical = F.conv2d(image, sobel_operator_vertical, padding=1, stride=1)

        # 计算边缘强度
        image_edges = torch.sqrt(image_horizontal ** 2 + image_vertical ** 2+1e-10)
        return image_edges

    def forward(self, F,image1):

        grad_img1 = self.LaplaceAlogrithm(image1, cuda_visible = True)
        # grad_img2 = self.SobelOperator(image2, cuda_visible = True)
        fus = self.LaplaceAlogrithm(F, cuda_visible = True)
        # y = torch.round((grad_img1+grad_img2)//torch.abs(grad_img1+grad_img2+0.0000000001)*torch.max( torch.abs(grad_img1), torch.abs(grad_img2)))
        # y = torch.max(grad_img1,grad_img2)
        loss = torch.norm(fus- grad_img1,p=1)
        return loss