import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math
import sys
import Utils
from Utils.CubePad import CustomPad


def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def e2c(equirectangular):
    cube = Utils.Equirec2Cube.ToCubeTensor(equirectangular.cuda())
    return cube

def c2e(cube):
    equirectangular = Utils.Cube2Equirec.ToEquirecTensor(cube.cuda())
    return equirectangular

class PreprocBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_lst, stride=2):
        super(PreprocBlock, self).__init__()
        assert len(kernel_size_lst) == 4 and out_channels % 4 == 0
        self.lst = nn.ModuleList([])

        for (h, w) in kernel_size_lst:
            padding = (h//2, w//2)
            tmp = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels//4, kernel_size=(h,w), stride=stride, padding=padding),
                        nn.BatchNorm2d(out_channels//4),
                        nn.ReLU(inplace=True)
                    )
            self.lst.append(tmp)

    def forward(self, x):
        out = []
        for conv in self.lst:
            out.append(conv(x))
        out = torch.cat(out, dim=1)
        return out

class fusion_ResNet(nn.Module):
    _output_size_init = (256, 256)

    def __init__(self, bs, layers, decoder, output_size=None, in_channels=3, pretrained=True, padding='ZeroPad'):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError(
                'Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(fusion_ResNet, self).__init__()
        self.padding = getattr(Utils.CubePad, padding)
        self.pad_7 = self.padding(3)
        self.pad_3 = self.padding(1)
        try: from . import resnet
        except: import resnet
        pretrained_model = getattr(resnet, 'resnet%d'%layers)(pretrained=pretrained, padding=padding)

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)

        self.output_size = output_size
        if output_size == None:
            output_size = _output_size_init
        else:
            assert isinstance(output_size, tuple)
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels, num_channels //
                               2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels//2)

        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        
        self.pre1 = PreprocBlock(3, 64, [[3, 9], [5, 11], [5, 7], [7, 7]])
        self.pre1.apply(weights_init)

    def forward(self, inputs):
        # resnet
        x = inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x0 = x
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.conv2(x4)
        x = self.bn2(x)

        return x

    def pre_encoder(self, x):
        x = self.conv1(self.pad_7(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(self.pad_3(x))

        return x
    
    # def pre_encoder2(self, x):
    #     x = self.pre1(x)
    #     x = self.maxpool(self.pad_3(x))

        return x

class CETransform(nn.Module):
    def __init__(self):
        super(CETransform, self).__init__()
        equ_h = [512, 128, 64, 32, 16]
        cube_h = [256, 64, 32, 16, 8]

        self.c2e = dict()
        self.e2c = dict()

        for h in equ_h:
            a = Utils.Equirec2Cube(1, h, h*2, h//2, 90)
            self.e2c['(%d,%d)' % (h, h*2)] = a

        for h in cube_h:
            a = Utils.Cube2Equirec(1, h, h*2, h*4)
            self.c2e['(%d)' % (h)] = a

    def E2C(self, x):
        [bs, c, h, w] = x.shape
        key = '(%d,%d)' % (h, w)
        assert key in self.e2c
        return self.e2c[key].ToCubeTensor(x)

    def C2E(self, x):
        [bs, c, h, w] = x.shape
        key = '(%d)' % (h)
        assert key in self.c2e and h == w
        return self.c2e[key].ToEquirecTensor(x)

    def forward(self, equi, cube):
        return self.e2c(equi), self.c2e(cube)




class MyModel(nn.Module):
    def __init__(self, layers, decoder, output_size=None, in_channels=3, pretrained=True):
        super(MyModel, self).__init__()
        bs = 1
        # self.equi_model = fusion_ResNet(
            # bs, layers, decoder, (512, 1024), 3, pretrained, padding='ZeroPad')


        #这里是指对CMP分支使用球形填充
        self.cube_model = fusion_ResNet(
            bs*6, layers, decoder, (256, 256), 3, pretrained, padding='SpherePad')
        
        #self.refine_model = Refine()

        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        #self.equi_decoder = choose_decoder(decoder, num_channels//2, padding='ZeroPad')
        # self.equi_conv3 = nn.Sequential(
        #         nn.Conv2d(num_channels//32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        #         nn.UpsamplingBilinear2d(size=(512, 1024))
        #         )
        #self.cube_decoder = choose_decoder(decoder, num_channels//2, padding='SpherePad')
        # mypad = getattr(Utils.CubePad, 'SpherePad')
        # self.cube_conv3 = nn.Sequential(
        #         mypad(1),
        #         nn.Conv2d(num_channels//32, 1, kernel_size=3, stride=1, padding=0, bias=False),
        #         nn.UpsamplingBilinear2d(size=(256, 256))
        #         )

        # self.equi_decoder.apply(weights_init)
        # self.equi_conv3.apply(weights_init)
        # self.cube_decoder.apply(weights_init)
        # self.cube_conv3.apply(weights_init)

        self.ce = CETransform()
        
        if layers <= 34:
            ch_lst = [64, 64, 128, 256, 512, 256, 128, 64, 32]
        else:
            ch_lst = [64, 256, 512, 1024, 2048, 1024, 512, 256, 128]

        # self.conv_e2c = nn.ModuleList([])
        # self.conv_c2e = nn.ModuleList([])
        # self.conv_mask = nn.ModuleList([])

        #下面仅供参考每个通道的大小 wilson
        # for i in range(9):
        #     conv_c2e = nn.Sequential(
        #                 nn.Conv2d(ch_lst[i], ch_lst[i], kernel_size=3, padding=1),
        #                 nn.ReLU(inplace=True)
        #             )
        #     conv_e2c = nn.Sequential(
        #                 nn.Conv2d(ch_lst[i], ch_lst[i], kernel_size=3, padding=1),
        #                 nn.ReLU(inplace=True)
        #             )
        #     conv_mask = nn.Sequential(
        #                 nn.Conv2d(ch_lst[i]*2, 1, kernel_size=1, padding=0),
        #                 nn.Sigmoid()
        #             )
        #     self.conv_e2c.append(conv_e2c)
        #     self.conv_c2e.append(conv_c2e)
        #     self.conv_mask.append(conv_mask)

        #self.grid = Utils.Equirec2Cube(None, 512, 1024, 256, 90).GetGrid()
        #self.d2p = Utils.Depth2Points(self.grid)

   
    def forward_FCRN_cube(self, cube):
        #这里还需要把每个阶段的特征提取出来，参考一下unifuse的代码怎么抽离出来
        #cube = self.ce.E2C(equi)
        feat_cube = self.cube_model.pre_encoder(cube)
        for e in range(5):
            if e < 4:
                feat_cube = getattr(self.cube_model, 'layer%d'%(e+1))(feat_cube)
            else:
                feat_cube = self.cube_model.conv2(feat_cube)  #这里的conv2就是为了调整通道数，通道减半 1x1卷积
                feat_cube = self.cube_model.bn2(feat_cube)
        # for d in range(4):
            # feat_cube = getattr(self.cube_decoder, 'layer%d'%(d+1))(feat_cube)
        # cube_depth = self.cube_conv3(feat_cube)
        return feat_cube



    def forward(self, x):
        return self.forward_FCRN_cube(x)

