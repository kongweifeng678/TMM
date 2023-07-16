from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from .resnet import *
from .mobilenet import *
from .layers import Conv3x3, ConvBlock, upsample, Cube2Equirec, Concat, BiProj, CEELayer
from . import spherical as S360
from collections import OrderedDict
import torch.nn.functional as F

class UniFuse_weight(nn.Module):
    """ UniFuse Model: Resnet based Euqi Encoder and Cube Encoder + Euqi Decoder
    """
    def __init__(self, num_layers, equi_h, equi_w, pretrained=False, max_depth=10.0,
                 fusion_type="cee", se_in_fusion=True):
        super(UniFuse_weight, self).__init__()

        self.num_layers = num_layers
        self.equi_h = equi_h
        self.equi_w = equi_w
        self.cube_h = equi_h//2

        self.fusion_type = fusion_type
        self.se_in_fusion = se_in_fusion

        # encoder
        encoder = {2: mobilenet_v2,
                   18: resnet18,
                   34: resnet34,
                   50: resnet50,
                   101: resnet101,
                   152: resnet152}

        if num_layers not in encoder:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))
        self.equi_encoder = encoder[num_layers](pretrained)
        self.cube_encoder = encoder[num_layers](pretrained)

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        if num_layers < 18:
            self.num_ch_enc = np.array([16, 24, 32, 96, 320])

        # decoder
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.equi_dec_convs = OrderedDict()
        self.c2e = {}

        Fusion_dict = {"cat": Concat,
                       "biproj": BiProj,
                       "cee": CEELayer}
        FusionLayer = Fusion_dict[self.fusion_type]


        self.c2e["5"] = Cube2Equirec(self.cube_h // 32, self.equi_h // 32, self.equi_w // 32)

        self.equi_dec_convs["fusion_5"] = FusionLayer(self.num_ch_enc[4], SE=self.se_in_fusion)
        self.equi_dec_convs["upconv_5"] = ConvBlock(self.num_ch_enc[4], self.num_ch_dec[4])

        self.c2e["4"] = Cube2Equirec(self.cube_h // 16, self.equi_h // 16, self.equi_w // 16)
        self.equi_dec_convs["fusion_4"] = FusionLayer(self.num_ch_enc[3], SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_4"] = ConvBlock(self.num_ch_dec[4] + self.num_ch_enc[3], self.num_ch_dec[4])
        self.equi_dec_convs["upconv_4"] = ConvBlock(self.num_ch_dec[4], self.num_ch_dec[3])

        self.c2e["3"] = Cube2Equirec(self.cube_h // 8, self.equi_h // 8, self.equi_w // 8)
        self.equi_dec_convs["fusion_3"] = FusionLayer(self.num_ch_enc[2], SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_3"] = ConvBlock(self.num_ch_dec[3] + self.num_ch_enc[2], self.num_ch_dec[3])
        self.equi_dec_convs["upconv_3"] = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2])

        self.c2e["2"] = Cube2Equirec(self.cube_h // 4, self.equi_h // 4, self.equi_w // 4)
        self.equi_dec_convs["fusion_2"] = FusionLayer(self.num_ch_enc[1], SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_2"] = ConvBlock(self.num_ch_dec[2] + self.num_ch_enc[1], self.num_ch_dec[2])
        self.equi_dec_convs["upconv_2"] = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1])

        self.c2e["1"] = Cube2Equirec(self.cube_h // 2, self.equi_h // 2, self.equi_w // 2)
        self.equi_dec_convs["fusion_1"] = FusionLayer(self.num_ch_enc[0], SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_1"] = ConvBlock(self.num_ch_dec[1] + self.num_ch_enc[0], self.num_ch_dec[1])
        self.equi_dec_convs["upconv_1"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0])

        self.equi_dec_convs["deconv_0"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])

        self.equi_dec_convs["depthconv_0"] = Conv3x3(self.num_ch_dec[0], 1)

        self.weight5 = Weight_upsample(self.num_ch_dec[4], self.equi_w // 16);
        self.weight4 = Weight_upsample(self.num_ch_dec[3], self.equi_w // 8);
        self.weight3 = Weight_upsample(self.num_ch_dec[2], self.equi_w // 4);
        self.weight2 = Weight_upsample(self.num_ch_dec[1], self.equi_w // 2);
        self.weight1 = Weight_upsample(self.num_ch_dec[0], self.equi_w);

        self.equi_decoder = nn.ModuleList(list(self.equi_dec_convs.values()))
        self.projectors = nn.ModuleList(list(self.c2e.values()))

        self.sigmoid = nn.Sigmoid()

        self.max_depth = nn.Parameter(torch.tensor(max_depth), requires_grad=False)

    def forward(self, input_equi_image, input_cube_image= torch.rand(1, 3, 128, 768).cuda()):
    #def forward(self, input_equi_image, input_cube_image):
        print(type(input_equi_image))
        print(type(input_cube_image))
        # euqi image encoding
    
        if self.num_layers < 18:
            equi_enc_feat0, equi_enc_feat1, equi_enc_feat2, equi_enc_feat3, equi_enc_feat4 \
                = self.equi_encoder(input_equi_image)
        else:
            x = self.equi_encoder.conv1(input_equi_image)
            x = self.equi_encoder.relu(self.equi_encoder.bn1(x))
            equi_enc_feat0 = x

            x = self.equi_encoder.maxpool(x)
            equi_enc_feat1 = self.equi_encoder.layer1(x)
            equi_enc_feat2 = self.equi_encoder.layer2(equi_enc_feat1)
            equi_enc_feat3 = self.equi_encoder.layer3(equi_enc_feat2)
            equi_enc_feat4 = self.equi_encoder.layer4(equi_enc_feat3)


        # cube image encoding
        cube_inputs = torch.cat(torch.split(input_cube_image, self.cube_h, dim=-1), dim=0)

        if self.num_layers < 18:
            cube_enc_feat0, cube_enc_feat1, cube_enc_feat2, cube_enc_feat3, cube_enc_feat4 \
                = self.cube_encoder(cube_inputs)
        else:

            x = self.cube_encoder.conv1(cube_inputs)
            x = self.cube_encoder.relu(self.cube_encoder.bn1(x))
            cube_enc_feat0 = x

            x = self.cube_encoder.maxpool(x)

            cube_enc_feat1 = self.cube_encoder.layer1(x)
            cube_enc_feat2 = self.cube_encoder.layer2(cube_enc_feat1)
            cube_enc_feat3 = self.cube_encoder.layer3(cube_enc_feat2)
            cube_enc_feat4 = self.cube_encoder.layer4(cube_enc_feat3)


        # euqi image decoding fused with cubemap features
        outputs = {}

        cube_enc_feat4 = torch.cat(torch.split(cube_enc_feat4, input_equi_image.shape[0], dim=0), dim=-1)
        c2e_enc_feat4 = self.c2e["5"](cube_enc_feat4)
        fused_feat4 = self.equi_dec_convs["fusion_5"](equi_enc_feat4, c2e_enc_feat4)
        equi_x = upsample(self.equi_dec_convs["upconv_5"](fused_feat4))
        equi_x = self.weight5(equi_x)

        cube_enc_feat3 = torch.cat(torch.split(cube_enc_feat3, input_equi_image.shape[0], dim=0), dim=-1)
        c2e_enc_feat3 = self.c2e["4"](cube_enc_feat3)
        fused_feat3 = self.equi_dec_convs["fusion_4"](equi_enc_feat3, c2e_enc_feat3)
        equi_x = torch.cat([equi_x, fused_feat3], 1)
        equi_x = self.equi_dec_convs["deconv_4"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_4"](equi_x))
        equi_x = self.weight4(equi_x)

        cube_enc_feat2 = torch.cat(torch.split(cube_enc_feat2, input_equi_image.shape[0], dim=0), dim=-1)
        c2e_enc_feat2 = self.c2e["3"](cube_enc_feat2)
        fused_feat2 = self.equi_dec_convs["fusion_3"](equi_enc_feat2, c2e_enc_feat2)
        equi_x = torch.cat([equi_x, fused_feat2], 1)
        equi_x = self.equi_dec_convs["deconv_3"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_3"](equi_x))
        equi_x = self.weight3(equi_x)

        cube_enc_feat1 = torch.cat(torch.split(cube_enc_feat1, input_equi_image.shape[0], dim=0), dim=-1)
        c2e_enc_feat1 = self.c2e["2"](cube_enc_feat1)
        fused_feat1 = self.equi_dec_convs["fusion_2"](equi_enc_feat1, c2e_enc_feat1)
        equi_x = torch.cat([equi_x, fused_feat1], 1)
        equi_x = self.equi_dec_convs["deconv_2"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_2"](equi_x))
        equi_x = self.weight2(equi_x)

        cube_enc_feat0 = torch.cat(torch.split(cube_enc_feat0, input_equi_image.shape[0], dim=0), dim=-1)
        c2e_enc_feat0 = self.c2e["1"](cube_enc_feat0)
        fused_feat0 = self.equi_dec_convs["fusion_1"](equi_enc_feat0, c2e_enc_feat0)
        equi_x = torch.cat([equi_x, fused_feat0], 1)
        equi_x = self.equi_dec_convs["deconv_1"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_1"](equi_x))
        equi_x = self.weight1(equi_x)

        equi_x = self.equi_dec_convs["deconv_0"](equi_x)

        equi_depth = self.equi_dec_convs["depthconv_0"](equi_x)
        outputs["pred_depth"] = self.max_depth * self.sigmoid(equi_depth)

        return outputs

class Weight_upsample(nn.Module):
    # Distortion-aware upsample module
    def __init__(self, in_channels, width):
        # 创建attention_map(尺寸)
        super(Weight_upsample, self).__init__()
        self.attention_weights = S360.weights.theta_confidence(
            S360.grid.create_spherical_grid(width)).cuda()
        # SEblock
        #self.seblock = SELayer(in_channels*2)
        self.esa = ESA(in_channels*2)
        # conv1x1 调整通道
        self.conv1x1 = nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels, \
            kernel_size=1, padding=0, stride=1)
    def forward(self, x):
        y = x * self.attention_weights
        fusion = torch.cat((x,y), dim=1)
        #fusion = self.seblock(fusion)
        fusion = self.esa(fusion)
        fusion = self.conv1x1(fusion)
        return fusion

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
        
def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
    if not padding and stride == 1:
        padding = kernel_size // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)

class ESA(nn.Module):
    def __init__(self, n_feats, conv=default_conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        f = x
        c1_ = (self.conv1(f))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m
