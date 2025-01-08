import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as tF
import torchvision.models as models

import numpy as np
import PIL
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

class JointToTensor(object):
    def __call__(self, img, target):
        return tF.to_tensor(img), tF.to_tensor(target)

class JointCenterCrop(object):
    def __init__(self, size):
        """
        Cut the image in the center with size size.

        params:
            size (int) : size of the center crop
        """
        self.size = size
        
    def __call__(self, img, target):
        return (tF.five_crop(img, self.size)[4], 
                tF.five_crop(target, self.size)[4])

class JointResizeCenterCrop(object):
    def __init__(self,size):
        """
        Cut the image in the center with size size. If the image is not big enough, first resize it.

        params:
            size (int) : size of the center crop
        """
        self.size = size
        
    def __call__(self, img, target):

        size = self.size

        if img.size[1] < size:
            scale = size / img.size[1] + 1
            new_height = int(img.size[1] * scale)
            new_width = int(img.size[0] * scale)
        
            img = tF.resize(img, (new_height, new_width))
            target = tF.resize(target, (new_height, new_width))
        
        if img.size[0] < size:
            scale = size / img.size[1] + 1
            new_height = int(img.size[1] * scale)
            new_width = int(img.size[0] * scale)
        
            img = tF.resize(img, (new_height, new_width))
            target = tF.resize(target, (new_height, new_width))
            
        return (tF.five_crop(img, self.size)[4], 
                tF.five_crop(target, self.size)[4])

class JointRandomResizeCrop(object):
    def __init__(self,size, minimum_scale = 0, maximum_scale = 1):
        """
        resize by scale between minimum_scale and maximum_scale, then crop a random location and resize.

        params:
            minimum_scale,maximum_scale : a random scale between [minimum_scale,maximum_scale] 
            size (int) : size of the center crop
        """
        self.size = size
        self.minimum_scale = minimum_scale
        self.maximum_scale = maximum_scale
        
    def __call__(self, img, target):
        #resize by scale between [minimum_scale,maximum_scale] 
              
        scale = torch.rand(1) * (self.maximum_scale - self.minimum_scale) + self.minimum_scale
        
        new_height = int(img.size[1] * scale)
        new_width = int(img.size[0] * scale)
        
        img = tF.resize(img, (new_height, new_width))
        target = tF.resize(target, (new_height, new_width))
        
        # crop a random location and resize
        
        crop_size = min(self.size, new_height, new_width)
        
        i = torch.randint(0, new_height - crop_size  + 1, size=(1,)).item()
        j = torch.randint(0, new_width - crop_size + 1, size=(1,)).item()
        
        img = tF.resized_crop(img, i ,j, crop_size, crop_size, size = self.size)
        target = tF.resized_crop(target, i ,j, crop_size, crop_size, size = self.size)
        
        return (img, target)

class JointNormalize(object):
    
    def __init__(self, mean, std):
        """
        normalize the image based on mean and std

        params: 
           mean, std : the mean and standard deviation needed.
        """

        self.mean = mean
        self.std = std
       
    def __call__(self, img, target):
                
        img = tF.normalize(img, self.mean, self.std)
        target = tF.normalize(target,self.mean, self.std)
        
        return (img, target)

class JointCompose(object):
    def __init__(self, transforms):
        """
        params: 
           transforms (list) : list of transforms
        """
        self.transforms = transforms

    # We override the __call__ function such that this class can be
    # called as a function i.e. JointCompose(transforms)(img, target)
    # Such classes are known as "functors"
    def __call__(self, img, target):
        """
        params:
            img (PIL.Image)    : input image
            target (PIL.Image) : ground truth label 
        """
        assert img.size == target.size
        for t in self.transforms:
            img, target = t(img, target)
        return img, target
    
def InvNormalize(img, mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.225]):
    """
    inverse the image from normalized version back to original image.
    """
    return tF.normalize(img ,mean=mean,std=std)

def fix_img(hole_img, pre_img, hole_size):
    """
    fix the hole in the center of hole_img by adding the center of the pre_img back.
    """
    h, w = hole_img.size
    mask = PIL.Image.new('1', hole_img.size, color=1)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((h//2 - hole_size//2, w//2 - hole_size//2, h//2 + hole_size//2,w//2 + hole_size//2), fill = 0)
    return PIL.Image.composite(hole_img, pre_img, mask)

def plot_4(sample_img, sample_target, sample_output_O, hole_size, invnorm = False):
    """
    plot 4 image
    1. sample_img, 2. sample_target, 
    3. the output of nerwork forward run and 4. the fixed image by combining the center of the forward run back to sample_img
    if invnorm is set the True, we first do inverse norm for image 1 and 2, image 3 and 4 will always be inverse normed
    """
    if invnorm:
        sample_img = InvNormalize(sample_img.cpu())
        sample_target = InvNormalize(sample_target.cpu())
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(2,2,1)
    plt.title('image sample')
    ax1.imshow(tF.to_pil_image(sample_img))
    ax2 = fig.add_subplot(2,2,2)
    plt.title('ground truth')
    ax2.imshow(tF.to_pil_image(sample_target))
    ax3 = fig.add_subplot(2,2,3)
    plt.title('prediction')
    pre_img = tF.to_pil_image(InvNormalize(sample_output_O.cpu()))
    ax3.imshow(pre_img)
    ax4 = fig.add_subplot(2,2,4)
    plt.title('compound result')
    ax4.imshow(fix_img(tF.to_pil_image(sample_img), pre_img, hole_size))

def plot_2(sample_img, sample_target, sample_output_O, hole_size, invnorm = False):
    """
    plot 2 image, 1. sample_target and 2. the fixed image by combining the center of the forward run back to sample_img
    if invnorm is set the True, we first do inverse norm for sample_target, image 2 will always be inverse normed
    """
    if invnorm:
        sample_img = InvNormalize(sample_img.cpu())
        sample_target = InvNormalize(sample_target.cpu())
    fig = plt.figure(figsize=(5, 3))
    ax1 = fig.add_subplot(1,2,1)
    plt.title('ground truth')
    ax1.imshow(tF.to_pil_image(sample_target))
    ax2 = fig.add_subplot(1,2,2)
    plt.title('compound result')
    pre_img = tF.to_pil_image(InvNormalize(sample_output_O.cpu()))
    ax2.imshow(fix_img(tF.to_pil_image(sample_img), pre_img, hole_size))

def plot_compare(sample_target1, sample_output_O1, sample_target2, sample_output_O2, hole_size, invnorm= False):
    """
    plot 4 image
    1. sample_target1, 2. the fixed image by combining the center of the forward run back to sample_img1, 
    3. sample_target2 and 4. the fixed image by combining the center of the forward run back to sample_img2
    if invnorm is set the True, we first do inverse norm for image 1 and 3, image 2 and 4 will always be inverse normed
    """
    if invnorm:
        sample_target1 = InvNormalize(sample_target1.cpu())
        sample_target2 = InvNormalize(sample_target2.cpu())
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(2,2,1)
    plt.title('ground truth')
    ax1.imshow(tF.to_pil_image(sample_target1))
    ax2 = fig.add_subplot(2,2,2)
    plt.title('compound result')
    pre_img1 = tF.to_pil_image(InvNormalize(sample_output_O1.cpu()))
    ax2.imshow(fix_img(tF.to_pil_image(sample_target1), pre_img1, hole_size))
    ax3 = fig.add_subplot(2,2,3)
    plt.title('ground truth')
    ax3.imshow(tF.to_pil_image(sample_target2))
    ax4 = fig.add_subplot(2,2,4)
    plt.title('compound result')
    pre_img2 = tF.to_pil_image(InvNormalize(sample_output_O2.cpu()))
    ax4.imshow(fix_img(tF.to_pil_image(sample_target2), pre_img2, hole_size))

