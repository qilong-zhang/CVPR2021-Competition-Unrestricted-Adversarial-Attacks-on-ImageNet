import os
import torch
import torchvision.models as models
from torch.autograd import Variable as V
from torch import nn
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision.datasets import ImageFolder
from Normalize import Normalize
from loader import ImageNet
from torch.utils.data import DataLoader
import argparse
import timm
import lpips

parser = argparse.ArgumentParser()
# parser.add_argument('--input_csv', type=str, default='', help='Input directory with images.')
# parser.add_argument('--input_dir', type=str, default='', help='Input directory with images.')
parser.add_argument('--mean', type=float, default=np.array([0.485, 0.456, 0.406]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.229, 0.224, 0.225]), help='std.')
# parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
# parser.add_argument("--num_iter_set", type=int, default=20, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=500, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=500, help="Height of each input images.")
parser.add_argument("--image_resize", type=int, default=560, help="Height of each input images.")
# parser.add_argument("--batch_size", type=int, default=10, help="How many images process at one time.")
# parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
# parser.add_argument("--amplification", type=float, default=10.0, help="To amplifythe step size.")
parser.add_argument("--prob", type=float, default=0.7, help="probability of using diverse inputs.")

opt = parser.parse_args()

def save_img(save_path, img):
    Image.fromarray(np.array(img * 255).astype('uint8')).save(save_path, quality=95)

def input_diversity(input_tensor):
    rnd = torch.randint(opt.image_width, opt.image_resize, ())
    rescaled = F.interpolate(input_tensor, size = [rnd, rnd], mode = 'bilinear', align_corners=True)
    h_rem = opt.image_resize - rnd
    w_rem = opt.image_resize - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    pad_list = (pad_left, pad_right, pad_top, pad_bottom)
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
    padded = nn.functional.interpolate(padded, [opt.image_resize, opt.image_resize])
    return padded if torch.rand(()) < opt.prob else input_tensor

def ensemble_input_diversity(input_tensor, idx):
    # [560,620,680,740,800] --> [575, 650, 725, 800]
    rnd = torch.randint(opt.image_width, [575, 650, 725, 800][idx], ())
    rescaled = F.interpolate(input_tensor, size = [rnd, rnd], mode = 'bilinear', align_corners=True)
    h_rem = [575, 650, 725, 800][idx] - rnd
    w_rem = [575, 650, 725, 800][idx] - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    pad_list = (pad_left, pad_right, pad_top, pad_bottom)
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
    padded = nn.functional.interpolate(padded, [256, 256], mode = 'bilinear')
    return padded


def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern])
    stack_kern = np.expand_dims(stack_kern, 1)
    stack_kern = torch.tensor(stack_kern).cuda()
    return stack_kern, kern_size // 2

def project_noise(x, stack_kern, kern_size):
    # x = tf.pad(x, [[0,0],[kern_size,kern_size],[kern_size,kern_size],[0,0]], "CONSTANT")
    x = F.conv2d(x, stack_kern, padding = (kern_size, kern_size), groups=3)
    return x

def torch_staircase_sign(noise, n):
    noise_staircase = torch.zeros(size=noise.shape).cuda()
    sign = torch.sign(noise).cuda()
    temp_noise = noise.cuda()
    abs_noise = abs(noise)
    base = n / 100
    percentile = []
    for i in np.arange(n, 100.1, n):
        percentile.append(i / 100.0)
    medium_now = torch.quantile(abs_noise.reshape(len(abs_noise), -1), q = torch.tensor(percentile, dtype=torch.float32).cuda(), dim = 1, keepdim = True).unsqueeze(2).unsqueeze(3)

    for j in range(len(medium_now)):
        # print(temp_noise.shape)
        # print(medium_now[j].shape)
        update = sign * (abs(temp_noise) <= medium_now[j]) * (base + 2 * base * j)
        noise_staircase += update
        temp_noise += update * 1e5

    return noise_staircase


