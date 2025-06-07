import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, vgg16_bn
from torchvision.models import resnet50
# from models.ssm import Block_mamba
from mamba_ssm import Mamba
from models.model import CoAttLayer, SAM
from models.model import ResBlk
from models.pvt import pvt_v2_b2
from config import Config
from einops import rearrange
from models.mamba2d import VSSBlock

# from models.transformer import LG

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.config = Config()
        bb = self.config.bb
        if bb == 'trans-pvt':
            self.bb = pvt_v2_b2()
            if self.config.pvt_weights:
                if os.path.exists(self.config.pvt_weights):
                    save_model = torch.load(self.config.pvt_weights)
                    model_dict = self.bb.state_dict()
                    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
                    model_dict.update(state_dict)
                    self.bb.load_state_dict(model_dict)
                else:
                    print("Warning: We cannot load the PVT backbone weights.")
                    print("\tIf you are testing/eval, it's okay.")
                    print("\tIf you are training, save it at {}.".format(self.config.pvt_weights))

        
        lateral_channels_in = {
            'trans-pvt': [512, 320, 128, 64],
        }
 
        self.conv320_512 = nn.Conv2d(320, 512, kernel_size=3, stride=1, padding=1)
        self.conv512_320 = nn.Conv2d(512, 320, kernel_size=3, stride=1, padding=1)
        self.conv128_64 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        if self.config.dec_blk == 'ResBlk':
            DecBlk = ResBlk

        self.top_layer = DecBlk(lateral_channels_in[bb][0], lateral_channels_in[bb][1])

        self.dec_layer4 = DecBlk(lateral_channels_in[bb][1], lateral_channels_in[bb][1])
        self.lat_layer4 = nn.Conv2d(lateral_channels_in[bb][1], lateral_channels_in[bb][1], 1, 1, 0)

        self.dec_layer3 = DecBlk(lateral_channels_in[bb][1], lateral_channels_in[bb][2])
        self.lat_layer3 = nn.Conv2d(lateral_channels_in[bb][2], lateral_channels_in[bb][2], 1, 1, 0)

        self.dec_layer2 = DecBlk(lateral_channels_in[bb][2], lateral_channels_in[bb][3])
        self.lat_layer2 = nn.Conv2d(lateral_channels_in[bb][3], lateral_channels_in[bb][3], 1, 1, 0)

        self.dec_layer1 = DecBlk(lateral_channels_in[bb][3], lateral_channels_in[bb][3] // 2)
        self.conv_out1 = nn.Sequential(nn.Conv2d(lateral_channels_in[bb][3] // 2, 1, 1, 1, 0))
        self.group = CoAttLayer()
        self.sam2 = SAM(64)
        self.sam1 = SAM(32)
        self.conv128_512 = nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1)
        self.conv512_64 = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1)
        self.conv64_512 = nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1)
        self.small_decoder = nn.Sequential(nn.Conv2d(512, 128, 3, stride=1, padding=1),
                                           nn.BatchNorm2d(128),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(128, 1, 1, stride=1, padding=0))

    def forward(self, x, gts):
        ########## Encoder ##########

        if 'trans' in self.config.bb:
            x1, x2, x3, x4 = self.bb(x)
        else:
            x1 = self.bb.conv1(x)
            x2 = self.bb.conv2(x1)
            x3 = self.bb.conv3(x2)
            x4 = self.bb.conv4(x3)
        # x1([24, 64, 64, 64])
        # x2([24, 128, 32, 32])
        # x3([24, 320, 16, 16])
        # x4([24, 512, 8, 8])
        
        if self.training:
            new_p, new_gt, new_n, new12, pred_contrast = self.group(x4, x3, gts)
        else:   
            new12 = self.group(x4, x3, gts)
            
        s1, s2 = torch.chunk(x2, 2, dim=1)
        x2 = self.sam2(s1, s2)
        t1, t2 = torch.chunk(x1, 2, dim=1)
        x1 = self.sam1(t1, t2)
        p4 = self.top_layer(new12)

        ########## Decoder ##########
        scaled_preds = []

        p4 = self.dec_layer4(p4)
        p4 = F.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        p3 = p4 + self.lat_layer4(x3)

        p3 = self.dec_layer3(p3)
        p3 = F.interpolate(p3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        p2 = p3 + self.lat_layer3(x2)

        p2 = self.dec_layer2(p2)
        p2 = F.interpolate(p2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        p1 = p2 + self.lat_layer2(x1)
        
        p1 = self.dec_layer1(p1)
        p1 = F.interpolate(p1, size=x.shape[2:], mode='bilinear', align_corners=True)
        if self.config.db_output_decoder:
            p1_out = self.db_output_decoder(p1)
        else:
            p1_out = self.conv_out1(p1)
        scaled_preds.append(p1_out)

        if self.training:
            return_values = [scaled_preds, new_p, new_gt, new_n, new12, pred_contrast]
            return return_values
        else:
            return scaled_preds
