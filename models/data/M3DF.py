import os
import sys
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from natsort import natsorted
import kornia.utils as KU
from scipy import io
from utils import util
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


def imsave(img, filename):
    img = img.squeeze().cpu()
    img = KU.tensor_to_image(img) * 255.
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

class M3DF(torch.utils.data.Dataset):
    """M3DF dataset for fusion.
    Args:
        root (str): The root directory of the dataset.
        train (bool): Whether to use the training set. Defaults to True.
        img_size (int): The size of the image. Defaults to 256.
        patch_size (int): The size of the patch. Defaults to 256.
        (ir_Y, vis_Y,vis_Cb, vis_Cr): (B, 1, H, W)
    """
    def __init__(self, root, train=True):
        super(M3DF, self).__init__()
        self.vis_folder = os.path.join(root, 'VIS/')
        self.ir_folder = os.path.join(root, 'IR/')

        self.ir_list = sorted(os.listdir(self.ir_folder))
        self.vis_list = sorted(os.listdir(self.vis_folder))
        # transform pic size to 256*256
        self.transforms = torchvision.transforms.Resize((256, 256))

        
    def print_info(self):
        print('path of vi:', self.vis_folder, 'path of ir:', self.ir_folder)
        print('number of vi:', len(self.vis_list),'number of ir:', len(self.ir_list))
        print('transforms size:', (self.transforms.size[0], self.transforms.size[1]))

    def __getitem__(self, index):
        vis_path = os.path.join(self.vis_folder, self.vis_list[index])
        ir_path = os.path.join(self.ir_folder, self.ir_list[index])
        vis = self.imread(path=vis_path)
        ir = self.imread(path=ir_path)
        vis_Y, vis_Cb, vis_Cr = util.RGB2YCrCb(vis)
        ir_Y, ir_Cb, ir_Cr = util.RGB2YCrCb(ir)
        vis_ir = torch.cat([vis_Y, ir_Y, vis_Cb, vis_Cr], dim=1)
        if vis_ir.shape[-1] <= 256 or vis_ir.shape[-2] <= 256:
            vis_ir = TF.resize(vis_ir, 256)
        patch = self.transforms(vis_ir)
        vis_Y, ir_Y, vis_Cb, vis_Cr = torch.split(patch, [1, 1, 1, 1], dim=1)
        return ir_Y.squeeze(0), vis_Y.squeeze(0), vis_Cb.squeeze(0), vis_Cr.squeeze(0), self.vis_list[index]

    def __len__(self):
        return len(self.vis_list)

    @staticmethod
    def imread(path):
        img = Image.open(path).convert('RGB')
        im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts

