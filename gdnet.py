import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from misc import crf_refine, check_mkdir
from dataset import ImageFolder, TestImageFolder
from backbone.resnext.resnext101_regular import ResNeXt101

import pytorch_lightning as pl
from utils.optimizer import get_optim
from PIL import Image
from skimage import feature

import wandb

from utils.loss import lovasz_hinge, edge_loss

to_pil = transforms.ToPILImage()

###################################################################
############################ CBAM #################################
###################################################################
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        # original
        # return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        # max
        # torch.max(x, 1)[0].unsqueeze(1)
        # avg
        return torch.mean(x, 1).unsqueeze(1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(1, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels=128, reduction_ratio=16, pool_types=['avg'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


###################################################################
############################ LCFI #################################
###################################################################
class LCFI(nn.Module):
    def __init__(self, input_channels, dr1=1, dr2=2, dr3=3, dr4=4):
        super(LCFI, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)
        self.channels_double = int(input_channels / 2)
        self.dr1 = dr1
        self.dr2 = dr2
        self.dr3 = dr3
        self.dr4 = dr4
        self.padding1 = 1 * dr1
        self.padding2 = 2 * dr2
        self.padding3 = 3 * dr3
        self.padding4 = 4 * dr4

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 3, 1, 1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 3, 1, 1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 3, 1, 1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 3, 1, 1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p1_d1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, (3, 1), 1, padding=(self.padding1, 0),
                      dilation=(self.dr1, 1)),
            nn.Conv2d(self.channels_single, self.channels_single, (1, 3), 1, padding=(0, self.padding1),
                      dilation=(1, self.dr1)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_d2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, (1, 3), 1, padding=(0, self.padding1),
                      dilation=(1, self.dr1)),
            nn.Conv2d(self.channels_single, self.channels_single, (3, 1), 1, padding=(self.padding1, 0),
                      dilation=(self.dr1, 1)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_fusion = nn.Sequential(nn.Conv2d(self.channels_double, self.channels_single, 3, 1, 1, dilation=1),
                                       nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p2_d1 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (5, 1), 1, padding=(self.padding2, 0),
                      dilation=(self.dr2, 1)),
            nn.Conv2d(self.channels_single, self.channels_single, (1, 5), 1, padding=(0, self.padding2),
                      dilation=(1, self.dr2)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_d2 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (1, 5), 1, padding=(0, self.padding2),
                      dilation=(1, self.dr2)),
            nn.Conv2d(self.channels_single, self.channels_single, (5, 1), 1, padding=(self.padding2, 0),
                      dilation=(self.dr2, 1)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_fusion = nn.Sequential(nn.Conv2d(self.channels_double, self.channels_single, 3, 1, 1, dilation=1),
                                       nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3_d1 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (7, 1), 1, padding=(self.padding3, 0),
                      dilation=(self.dr3, 1)),
            nn.Conv2d(self.channels_single, self.channels_single, (1, 7), 1, padding=(0, self.padding3),
                      dilation=(1, self.dr3)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_d2 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (1, 7), 1, padding=(0, self.padding3),
                      dilation=(1, self.dr3)),
            nn.Conv2d(self.channels_single, self.channels_single, (7, 1), 1, padding=(self.padding3, 0),
                      dilation=(self.dr3, 1)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_fusion = nn.Sequential(nn.Conv2d(self.channels_double, self.channels_single, 3, 1, 1, dilation=1),
                                       nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p4_d1 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (9, 1), 1, padding=(self.padding4, 0),
                      dilation=(self.dr4, 1)),
            nn.Conv2d(self.channels_single, self.channels_single, (1, 9), 1, padding=(0, self.padding4),
                      dilation=(1, self.dr4)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_d2 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (1, 9), 1, padding=(0, self.padding4),
                      dilation=(1, self.dr4)),
            nn.Conv2d(self.channels_single, self.channels_single, (9, 1), 1, padding=(self.padding4, 0),
                      dilation=(self.dr4, 1)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_fusion = nn.Sequential(nn.Conv2d(self.channels_double, self.channels_single, 3, 1, 1, dilation=1),
                                       nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.cbam = CBAM(self.input_channels)

        self.channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 3, 1, 1, dilation=1),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1_fusion(torch.cat((self.p1_d1(p1_input), self.p1_d2(p1_input)), 1))

        p2_input = torch.cat((self.p2_channel_reduction(x), p1), 1)
        p2 = self.p2_fusion(torch.cat((self.p2_d1(p2_input), self.p2_d2(p2_input)), 1))

        p3_input = torch.cat((self.p3_channel_reduction(x), p2), 1)
        p3 = self.p3_fusion(torch.cat((self.p3_d1(p3_input), self.p3_d2(p3_input)), 1))

        p4_input = torch.cat((self.p4_channel_reduction(x), p3), 1)
        p4 = self.p4_fusion(torch.cat((self.p4_d1(p4_input), self.p4_d2(p4_input)), 1))

        channel_reduction = self.channel_reduction(self.cbam(torch.cat((p1, p2, p3, p4), 1)))

        return channel_reduction


###################################################################
############################ NETWORK ##############################
###################################################################
class GDNet(nn.Module):
    def __init__(self, backbone_path=None):
        super(GDNet, self).__init__()
        # params

        # backbone
        resnext = ResNeXt101(backbone_path)
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.h5_conv = LCFI(2048, 1, 2, 3, 4)
        self.h4_conv = LCFI(1024, 1, 2, 3, 4)
        self.h3_conv = LCFI(512, 1, 2, 3, 4)
        self.l2_conv = LCFI(256, 1, 2, 3, 4)

        # h fusion
        self.h5_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.h3_down = nn.AvgPool2d((2, 2), stride=2)
        self.h_fusion = CBAM(896)
        self.h_fusion_conv = nn.Sequential(nn.Conv2d(896, 896, 3, 1, 1), nn.BatchNorm2d(896), nn.ReLU())

        # l fusion
        self.l_fusion_conv = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.h2l = nn.ConvTranspose2d(896, 1, 8, 4, 2)

        # final fusion
        self.h_up_for_final_fusion = nn.ConvTranspose2d(896, 256, 8, 4, 2)
        self.final_fusion = CBAM(320)
        self.final_fusion_conv = nn.Sequential(nn.Conv2d(320, 320, 3, 1, 1), nn.BatchNorm2d(320), nn.ReLU())

        # predict conv
        self.h_predict = nn.Conv2d(896, 1, 3, 1, 1)
        self.l_predict = nn.Conv2d(64, 1, 3, 1, 1)
        self.final_predict = nn.Conv2d(320, 1, 3, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        # x: [batch_size, channel=3, h, w]
        
        layer0 = self.layer0(x)  # [-1, 64, h/2, w/2]
        layer1 = self.layer1(layer0)  # [-1, 256, h/4, w/4]
        layer2 = self.layer2(layer1)  # [-1, 512, h/8, w/8]
        layer3 = self.layer3(layer2)  # [-1, 1024, h/16, w/16]
        layer4 = self.layer4(layer3)  # [-1, 2048, h/32, w/32]

        h5_conv = self.h5_conv(layer4)
        h4_conv = self.h4_conv(layer3)
        h3_conv = self.h3_conv(layer2)
        l2_conv = self.l2_conv(layer1)


        # h fusion
        h5_up = self.h5_up(h5_conv)
        h3_down = self.h3_down(h3_conv)
        h_fusion = self.h_fusion(torch.cat((h5_up, h4_conv, h3_down), 1))
        h_fusion = self.h_fusion_conv(h_fusion)

        # l fusion
        l_fusion = self.l_fusion_conv(l2_conv)
        h2l = self.h2l(h_fusion)
        l_fusion = torch.sigmoid(h2l) * l_fusion

        # final fusion
        h_up_for_final_fusion = self.h_up_for_final_fusion(h_fusion)
        final_fusion = self.final_fusion(torch.cat((h_up_for_final_fusion, l_fusion), 1))
        final_fusion = self.final_fusion_conv(final_fusion)

        # h predict
        h_predict = self.h_predict(h_fusion)

        # l predict
        l_predict = self.l_predict(l_fusion)

        # final predict
        final_predict = self.final_predict(final_fusion)

        # rescale to original size
        h_predict = F.interpolate(h_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        l_predict = F.interpolate(l_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        final_predict = F.interpolate(final_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        return torch.sigmoid(h_predict), torch.sigmoid(l_predict), torch.sigmoid(final_predict)


###################################################################
# ###################### LIGHTNINH NETWORK ########################
###################################################################

class LitGDNet(pl.LightningModule):
    def __init__(self, args, backbone_path=None):
        super(LitGDNet, self).__init__()
        # params
        self.save_hyperparameters(args)
        self.val_iter = 0
        # backbone
        self.args = args
        self.sum_w_losses = sum(args.w_losses)
        self.testing_path = args.gdd_testing_root
        self.training_path = args.gdd_training_root
        self.eval_path = args.gdd_eval_root

        # Defining the metrics 
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

        # self.test_F1 = pl.metrics.F1(num_classes=1)

        # self.train_FBeta = pl.metrics.FBeta(num_classes=1, beta=0.5)
        # self.val_FBeta = pl.metrics.FBeta(num_classes=1, beta=0.5)
        # self.test_FBeta = pl.metrics.FBeta(num_classes=1, beta=0.5)

        resnext = ResNeXt101(backbone_path)
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        if self.args.freeze_resnet:
            self.layer0.eval()
            self.layer1.eval()
            self.layer2.eval()
            self.layer3.eval()
            self.layer4.eval()
            for child in self.layer0.children():
                child.requires_grad = False
            for child in self.layer1.children():
                child.requires_grad = False


        self.h5_conv = LCFI(2048, 1, 2, 3, 4)
        self.h4_conv = LCFI(1024, 1, 2, 3, 4)
        self.h3_conv = LCFI(512, 1, 2, 3, 4)
        self.l2_conv = LCFI(256, 1, 2, 3, 4)
        self.edge_conv_l = LCFI(256, 1, 2, 3, 4)
        self.edge_conv_h = LCFI(1024, 1, 2, 3, 4)

        if self.args.freeze_LCFI:
            self.h5_conv.eval()
            self.h4_conv.eval()
            self.h3_conv.eval()
            self.l2_conv.eval()
            self.edge_conv_l.eval()
            self.edge_conv_h.eval()

        # h fusion
        self.h5_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.h3_down = nn.AvgPool2d((2, 2), stride=2)
        self.h_fusion = CBAM(896)
        self.h_fusion_conv = nn.Sequential(nn.Conv2d(896, 896, 3, 1, 1), nn.BatchNorm2d(896), nn.ReLU())

        # edge fusion
        self.edge_h_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.edge_l_down = nn.AvgPool2d((2, 2), stride=2)
        self.edge_fusion = CBAM(320)
        self.edge_fusion_conv = nn.Sequential(nn.Conv2d(320, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 1, 3, 1, 1))

        # l fusion
        self.l_fusion_conv = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.h2l = nn.ConvTranspose2d(896, 1, 8, 4, 2)

        # final fusion
        self.h_up_for_final_fusion = nn.ConvTranspose2d(896, 256, 8, 4, 2)
        self.final_fusion = CBAM(320)
        self.final_fusion_conv = nn.Sequential(nn.Conv2d(320, 320, 3, 1, 1), nn.BatchNorm2d(320), nn.ReLU())

        # predict conv
        self.h_predict = nn.Conv2d(896, 1, 3, 1, 1)
        self.l_predict = nn.Conv2d(64, 1, 3, 1, 1)
        self.final_predict = nn.Conv2d(320, 1, 3, 1, 1)

        # Refinement
        self.refinement = nn.Sequential(nn.Conv2d(2, 1, 3, 1, 1), nn.Conv2d(1, 1, 3, 1, 1), nn.BatchNorm2d(1), nn.ReLU())

        # self.h5_up.eval()
        # self.h3_down.eval()
        # self.h_fusion.eval()
        # self.h_fusion_conv.eval()
        # self.l_fusion_conv.eval()
        # self.h2l.eval()

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

        ###############################
        # Defining the transoformations
        self.img_transform = transforms.Compose([
            transforms.Resize((self.args.scale, self.args.scale)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.args.scale, self.args.scale)),
            transforms.ToTensor()
        ])

    def forward(self, x):
        # x: [batch_size, channel=3, h, w]
        if self.args.freeze_resnet:
            self.layer0.eval()
            self.layer1.eval()
            self.layer2.eval()
            self.layer3.eval()
            self.layer4.eval()
            with torch.no_grad():
                layer0 = self.layer0(x)  # [-1, 64, h/2, w/2]
                layer1 = self.layer1(layer0)  # [-1, 256, h/4, w/4]
                layer2 = self.layer2(layer1)  # [-1, 512, h/8, w/8]
                layer3 = self.layer3(layer2)  # [-1, 1024, h/16, w/16]
                layer4 = self.layer4(layer3)  # [-1, 2048, h/32, w/32]
        else:
            layer0 = self.layer0(x)  # [-1, 64, h/2, w/2]
            layer1 = self.layer1(layer0)  # [-1, 256, h/4, w/4]
            layer2 = self.layer2(layer1)  # [-1, 512, h/8, w/8]
            layer3 = self.layer3(layer2)  # [-1, 1024, h/16, w/16]
            layer4 = self.layer4(layer3)  # [-1, 2048, h/32, w/32]

        if self.args.freeze_LCFI:
            self.h5_conv.eval()
            self.h4_conv.eval()
            self.h3_conv.eval()
            self.l2_conv.eval()
            self.edge_conv_l.eval()
            self.edge_conv_h.eval()
            with torch.no_grad():
                h5_conv = self.h5_conv(layer4)
                h4_conv = self.h4_conv(layer3)
                h3_conv = self.h3_conv(layer2)
                l2_conv = self.l2_conv(layer1)
                edge_conv_l = self.edge_conv_l(layer1)
                edge_conv_h = self.edge_conv_h(layer3)
        else:
            h5_conv = self.h5_conv(layer4)
            h4_conv = self.h4_conv(layer3)
            h3_conv = self.h3_conv(layer2)
            l2_conv = self.l2_conv(layer1)
            edge_conv_l = self.edge_conv_l(layer1)
            edge_conv_h = self.edge_conv_h(layer3)


        # h fusion
        h5_up = self.h5_up(h5_conv)
        h3_down = self.h3_down(h3_conv)
        h_fusion = self.h_fusion(torch.cat((h5_up, h4_conv, h3_down), 1))
        h_fusion = self.h_fusion_conv(h_fusion)

        # l fusion
        l_fusion = self.l_fusion_conv(l2_conv)
        h2l = self.h2l(h_fusion)
        l_fusion = torch.sigmoid(h2l) * l_fusion

        edge_h_up = self.edge_h_up(edge_conv_h)
        edge_l_down = self.edge_l_down(edge_conv_l)
        edge_fusion = self.edge_fusion(torch.cat((edge_h_up, edge_l_down), 1))
        edge_fusion = self.edge_fusion_conv(edge_fusion)


        # final fusion
        h_up_for_final_fusion = self.h_up_for_final_fusion(h_fusion)
        final_fusion = self.final_fusion(torch.cat((h_up_for_final_fusion, l_fusion), 1))
        final_fusion = self.final_fusion_conv(final_fusion)

        # h predict
        h_predict = self.h_predict(h_fusion)

        # l predict
        l_predict = self.l_predict(l_fusion)

        # final predict
        final_predict = self.final_predict(final_fusion)

        # rescale to original size
        h_predict = F.interpolate(h_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        l_predict = F.interpolate(l_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        final_predict = F.interpolate(final_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        edge_predict = F.interpolate(edge_fusion, size=x.size()[2:], mode='bilinear', align_corners=True)


        # final refined prediction
        refine_predict = self.refinement(torch.cat((edge_predict, final_predict), 1))

        return torch.sigmoid(h_predict), torch.sigmoid(l_predict), torch.sigmoid(final_predict), torch.sigmoid(refine_predict), torch.sigmoid(edge_predict)

    def calc_loss(self, f_1, f_2, f_3, outputs):
        loss = torch.tensor(0.)
        # loss = torch.tensor.new_zeros(1)
        # loss.requires_grad = True
        if f_1.is_cuda:
            loss = torch.tensor(0., device=f_1.device.index)
        loss_fun_w_sum = 1e-8
        if "lovasz" in self.args.loss_funcs:
            loss1 = lovasz_hinge(f_1, outputs, per_image=False)*self.args.w_losses[0]
            loss2 = lovasz_hinge(f_2, outputs, per_image=False)*self.args.w_losses[1]
            loss3 = lovasz_hinge(f_3, outputs, per_image=False)*self.args.w_losses[2]
            loss += self.args.w_losses_function[0]*(loss1 + loss2 + loss3)/self.sum_w_losses
            loss_fun_w_sum+= self.args.w_losses_function[0]

        if "BCE" in self.args.loss_funcs:
            loss_BCE_1 = F.binary_cross_entropy_with_logits(f_1, outputs)*self.args.w_losses[0]
            loss_BCE_2 = F.binary_cross_entropy_with_logits(f_2, outputs)*self.args.w_losses[1]
            loss_BCE_3 = F.binary_cross_entropy_with_logits(f_3, outputs)*self.args.w_losses[2]
            loss += self.args.w_losses_function[1]*(loss_BCE_1 + loss_BCE_2 + loss_BCE_3)/self.sum_w_losses
            loss_fun_w_sum+= self.args.w_losses_function[1]

        loss = loss/loss_fun_w_sum
        return loss

    def calc_loss_2(self, pred_h, pred_l, pred_f, pred_ref, pred_edges, outputs, edges):
        loss = torch.tensor(0.)
        # loss = torch.tensor.new_zeros(1)
        # loss.requires_grad = True
        if pred_h.is_cuda:
            loss = torch.tensor(0., device=pred_h.device.index)
        loss_fun_w_sum = 1e-8
        if "lovasz" in self.args.loss_funcs:
            loss1 = lovasz_hinge(pred_h, outputs, per_image=False)*self.args.w_losses[0]
            loss2 = lovasz_hinge(pred_l, outputs, per_image=False)*self.args.w_losses[1]
            loss3 = lovasz_hinge(pred_f, outputs, per_image=False)*self.args.w_losses[2]
            loss4 = lovasz_hinge(pred_ref, outputs, per_image=False)*self.args.w_losses[3]
            sum_w = self.args.w_losses[0] + self.args.w_losses[1] + self.args.w_losses[2] + self.args.w_losses[3]
            loss += self.args.w_losses_function[0]*(loss1 + loss2 + loss3 + loss4)/sum_w
            # loss += self.args.w_losses_function[0]*(loss1 + loss2 + loss3 + loss4)/self.sum_w_losses
            loss_fun_w_sum+= self.args.w_losses_function[0]

        if "BCE" in self.args.loss_funcs:
            loss_BCE_1 = F.binary_cross_entropy_with_logits(pred_h, outputs)*self.args.w_losses[0]
            loss_BCE_2 = F.binary_cross_entropy_with_logits(pred_l, outputs)*self.args.w_losses[1]
            loss_BCE_3 = F.binary_cross_entropy_with_logits(pred_f, outputs)*self.args.w_losses[2]
            loss_BCE_4 = F.binary_cross_entropy_with_logits(pred_ref, outputs)*self.args.w_losses[3]
            sum_w = self.args.w_losses[0] + self.args.w_losses[1] + self.args.w_losses[2] + self.args.w_losses[3]
            # loss_BCE_edge = F.binary_cross_entropy_with_logits(pred_edges, edges)*self.args.w_losses[4]
            loss += self.args.w_losses_function[1]*(loss_BCE_1 + loss_BCE_2 + loss_BCE_3 + loss_BCE_4)/sum_w
            loss_fun_w_sum+= self.args.w_losses_function[1]

        if "edge" in self.args.loss_funcs:
            edge_loss_val = edge_loss(pred_edges, edges)
            loss += self.args.w_losses_function[1]*edge_loss_val
            loss_fun_w_sum+= self.args.w_losses_function[2]
        loss = loss/loss_fun_w_sum
        return loss

    #####################################
    # Ligtning functions
    #####################################
    def configure_optimizers(self):
        optimizer = get_optim(self, self.args)
        return optimizer

    #####################################
    # Loader functions
    #####################################
    def train_dataloader(self):
        dataset = ImageFolder(self.training_path, img_transform= self.img_transform, target_transform= self.mask_transform, random_percent=self.args.rand_augmentation_prob, image_crop_percentage=self.args.image_crop_percentage)
        loader = DataLoader(dataset, batch_size= self.args.batch_size, num_workers = 4, shuffle=self.args.shuffle_dataset)

        return loader

    def val_dataloader(self):
        eval_dataset = ImageFolder(self.eval_path, img_transform= self.img_transform, target_transform= self.mask_transform)
        loader = DataLoader(eval_dataset, batch_size= self.args.eval_batch_size, num_workers = 4, shuffle=False)
        self.eval_set = eval_dataset
        
        return loader

    def test_dataloader(self):
        test_dataset = TestImageFolder(self.testing_path, img_transform= self.img_transform, target_transform= self.mask_transform, add_real_imgs = (self.args.developer_mode and not (self.args.mode == "train")))
        loader = DataLoader(test_dataset, batch_size= self.args.test_batch_size, num_workers = 4, shuffle=False)

        return loader

    def training_step(self, batch, batch_idx):
        inputs = batch[0]
        outputs = batch[1]
        edges = batch[2]

        inputs.requires_grad=True
        outputs.requires_grad=True
        # pred_h, pred_l, pred_f = self(inputs)
        pred_h, pred_l, pred_f, pred_ref, pred_edges = self(inputs)

        loss = self.calc_loss_2(pred_f, pred_l, pred_h, pred_ref, pred_edges, outputs, edges)
        # loss = self.calc_loss(pred_f, pred_l, pred_h, outputs)
        # loss1 = lovasz_hinge(pred_f, outputs, per_image=False)*self.args.w_losses[0]
        # loss2 = lovasz_hinge(pred_l, outputs, per_image=False)*self.args.w_losses[1]
        # loss3 = lovasz_hinge(pred_h, outputs, per_image=False)*self.args.w_losses[2]
        # loss = (loss1 + loss2 + loss3)/self.sum_w_losses


        #################################
        # Logging metrics
        #################################
        thresh = self.args.iou_threshold
        
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        self.train_acc(pred_f, outputs)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        
        if "F1" in self.args.metric_log:
            train_F1 = pl.metrics.functional.f1(pred_f, outputs, num_classes=1, threshold=thresh)
            self.log('train_F1', train_F1, on_step=False, on_epoch=True)
        
        if "FBeta" in self.args.metric_log:
            train_FBeta = pl.metrics.functional.fbeta(pred_f, outputs, num_classes=1, beta=0.5, threshold=thresh)
            self.log('train_FBeta', train_FBeta, on_step=False, on_epoch=True)
        
        if "precision" in self.args.metric_log:
            train_avg_precision = pl.metrics.functional.average_precision(pred_f, outputs, pos_label=1)
            self.log('train_avg_precision', train_avg_precision, on_step=False, on_epoch=True)

        if "recall" in self.args.metric_log:
            train_recall = pl.metrics.functional.classification.recall(pred_f, outputs)
            self.log('train_recall', train_recall, on_step=False, on_epoch=True)

        if "iou" in self.args.metric_log:
            t = torch.tensor(thresh)
            out_int = outputs.int()
            res_int = (pred_f>t).int()
            train_iou = pl.metrics.functional.classification.iou(res_int, out_int)
            self.log('train_iou', train_iou, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch[0]
        outputs = batch[1]
        edges = batch[2]
        input_size = inputs.shape
        input_size = list(input_size)

        inputs.requires_grad=False
        outputs.requires_grad=False
        
        pred_h, pred_l, pred_f, pred_ref, pred_edges = self(inputs)

        loss = self.calc_loss_2(pred_f, pred_l, pred_h, pred_ref, pred_edges, outputs, edges)

        # loss = self.calc_loss(pred_f, pred_l, pred_h, outputs)
        # loss1 = lovasz_hinge(pred_f, outputs, per_image=False)*self.args.w_losses[0]
        # loss2 = lovasz_hinge(pred_l, outputs, per_image=False)*self.args.w_losses[1]
        # loss3 = lovasz_hinge(pred_h, outputs, per_image=False)*self.args.w_losses[2]
        # loss = (loss1 + loss2 + loss3)/self.sum_w_losses

        #################################
        # Logging metrics
        #################################
        thresh = self.args.iou_threshold

        self.log('val_loss', loss, on_epoch=True)

        self.val_acc(pred_f, outputs)
        self.log('val_acc', self.val_acc, on_epoch=True)

        if "F1" in self.args.metric_log:
            val_F1 = pl.metrics.functional.f1(pred_f, outputs, num_classes=1, threshold=thresh)
            self.log('val_F1', val_F1, on_epoch=True)
        
        if "FBeta" in self.args.metric_log:
            val_FBeta = pl.metrics.functional.fbeta(pred_f, outputs, num_classes=1, beta=0.5, threshold=thresh)
            self.log('val_FBeta', val_FBeta, on_epoch=True)
        
        if "precision" in self.args.metric_log:
            val_avg_precision = pl.metrics.functional.average_precision(pred_f, outputs, pos_label=1)
            self.log('val_avg_precision', val_avg_precision, on_epoch=True)

        if "recall" in self.args.metric_log:
            val_recall = pl.metrics.functional.classification.recall(pred_f, outputs)
            self.log('val_recall', val_recall, on_epoch=True)

        if "iou" in self.args.metric_log:
            t = torch.tensor(self.args.iou_threshold)
            out_int = outputs.int()
            res_int = (pred_f>t).int()
            val_iou = pl.metrics.functional.classification.iou(res_int, out_int)
            self.log('val_iou', val_iou, on_epoch=True)

        return {'val_loss': loss, 'val_acc': self.val_acc, 'input_size': input_size}

    # Function which is activated when the validation epoch wnds
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        input_size = outputs[0]['input_size']
        input_size = [x[0] for x in input_size]
        # avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        # Array of images to log
        wandb_images = []
        for i in range(self.args.num_log_img):
            batch = self.eval_set.sample(1)
            inputs = batch["img"]
            outputs = batch["mask"]
            inputs = torch.from_numpy(inputs)
            outputs = torch.tensor(outputs)
            if len(self.args.device_ids) > 0:
                inputs = inputs.cuda(self.args.device_ids[0])
                outputs = outputs.cuda(self.args.device_ids[0])
            pred_h, pred_l, pred_f, pred_ref, pred_edges = self(inputs)
            f_1 = pred_f.data.cpu()
            rev_size = [batch["size"][0][1], batch["size"][0][0]]
            image1_size = batch["size"][0]
            f_1_trans = np.array(transforms.Resize(rev_size)(to_pil(f_1[0])))
            f_1_crf = crf_refine(np.array(batch["r_img"][0]), f_1_trans)

            f_2 = pred_l.data.cpu()
            f_2_trans = np.array(transforms.Resize(rev_size)(to_pil(f_2[0])))
            f_2_crf = crf_refine(np.array(batch["r_img"][0]), f_2_trans)

            f_3 = pred_h.data.cpu()
            f_3_trans = np.array(transforms.Resize(rev_size)(to_pil(f_3[0])))
            f_3_crf = crf_refine(np.array(batch["r_img"][0]), f_3_trans)
            
            f_ref = pred_ref.data.cpu()
            f_ref_trans = np.array(transforms.Resize(rev_size)(to_pil(f_ref[0])))
            f_ref_crf = crf_refine(np.array(batch["r_img"][0]), f_ref_trans)
            
            f_edges = pred_edges.data.cpu()
            f_edges_trans = np.array(transforms.Resize(rev_size)(to_pil(f_edges[0])))
            f_edges_crf = crf_refine(np.array(batch["r_img"][0]), f_edges_trans)
            
            img_res = Image.fromarray(f_1_crf)
            img_res2 = Image.fromarray(f_2_crf)
            img_res3 = Image.fromarray(f_3_crf)
            img_resref = Image.fromarray(f_ref_crf)
            img_resedges = Image.fromarray(f_edges_crf)

            real_image = np.array(batch["r_img"][0])
            real_mask = 255-np.array(batch["r_mask"][0])
            img_res_1 = 255-np.array(f_1_crf)
            img_res_2 = 255-np.array(f_2_crf)
            img_res_3 = 255-np.array(f_3_crf)
            img_res_ref = 255-np.array(f_ref_crf)
            img_res_edges = 255-np.array(f_edges_crf)

            class_labels = {0:"glass"}

            mask_img = wandb.Image(real_image, masks={
                "ground_truth":{
                    "mask_data": real_mask,
                    "class_labels": class_labels
                },
                "prediction": {
                    "mask_data": img_res_1,
                    "class_labels": class_labels
                },
                "l_prediction": {
                    "mask_data": img_res_2,
                    "class_labels": class_labels
                },
                "h_prediction": {
                    "mask_data": img_res_3,
                    "class_labels": class_labels
                },
                "ref_prediction": {
                    "mask_data": img_res_ref,
                    "class_labels": class_labels
                },
                "edge_prediction": {
                    "mask_data": img_res_edges,
                    "class_labels": class_labels
                }
            })
            wandb_images.append(mask_img)
            
            if self.args.developer_mode:

                new_image = Image.new('RGB',(3*image1_size[0], image1_size[1]), (250,250,250))
                new_image.paste(batch["r_img"][0],(0,0))
                new_image.paste(batch["r_mask"][0],(image1_size[0],0))
                new_image.paste(img_res,(image1_size[0]*2,0))

                new_image.save(os.path.join(self.args.gdd_results_root, self.args.log_name,
                                                        "Eval_Epoch: " + str(self.val_iter) +"_"+ str(i) +"_Pred.png"))

                new_image = Image.new('RGB',(3*image1_size[0], image1_size[1]), (250,250,250))
                new_image.paste(batch["r_mask"][0],(0,0))
                new_image.paste(img_res2,(image1_size[0],0))
                new_image.paste(img_res3,(image1_size[0]*2,0))

                new_image.save(os.path.join(self.args.gdd_results_root, self.args.log_name,
                                                        "Eval_Epoch: " + str(self.val_iter) +"_"+ str(i) +"_h_l_Pred.png"))

                new_image = Image.new('RGB',(3*image1_size[0], image1_size[1]), (250,250,250))
                new_image.paste(batch["r_edges"][0],(0,0))
                new_image.paste(img_resref,(image1_size[0],0))
                new_image.paste(img_resedges,(image1_size[0]*2,0))

                new_image.save(os.path.join(self.args.gdd_results_root, self.args.log_name,
                                                        "Eval_Epoch: " + str(self.val_iter) +"_"+ str(i) +"_ref_edg_Pred.png"))

        # The number of validation itteration
        self.val_iter +=1 

        model_filename = f"model_{str(self.global_step).zfill(5)}.onnx"
        if self.args.wandb:
            wandb.log({"examples_"+ str(self.val_iter): wandb_images})
            wandb.save(model_filename)


    def test_step(self, batch, batch_idx):
        inputs = batch[0]
        outputs = batch[1]
        real_inputs = batch[2].cpu().detach().numpy()
        real_outputs = batch[3].cpu().detach().numpy()
        sizes = batch[4].cpu().detach().numpy()
        input_size = inputs.shape
        input_size = list(input_size)

        inputs.requires_grad=False
        outputs.requires_grad=False
        
        pred_h, pred_l, pred_f, pred_ref, pred_edges = self(inputs)

        loss = self.calc_loss(pred_f, pred_l, pred_h, outputs)
        # loss1 = lovasz_hinge(pred_f, outputs, per_image=False)*self.args.w_losses[0]
        # loss2 = lovasz_hinge(pred_l, outputs, per_image=False)*self.args.w_losses[1]
        # loss3 = lovasz_hinge(pred_h, outputs, per_image=False)*self.args.w_losses[2]
        # loss = (loss1 + loss2 + loss3)/self.sum_w_losses

        #################################
        # Logging metrics
        #################################
        thresh = self.args.iou_threshold

        self.log('test_loss', loss, on_epoch=True)

        # self.test_acc(pred_f, outputs)
        # self.log('test_acc', self.test_acc, on_epoch=True)

        if "F1" in self.args.metric_log:
            test_F1 = pl.metrics.functional.f1(pred_f, outputs, num_classes=1, threshold=thresh)
            self.log('test_F1', test_F1, on_epoch=True)
        
        if "FBeta" in self.args.metric_log:
            test_FBeta = pl.metrics.functional.fbeta(pred_f, outputs, num_classes=1, beta=0.5, threshold=thresh)
            self.log('test_FBeta', test_FBeta, on_epoch=True)

        if "precision" in self.args.metric_log:
            test_avg_precision = pl.metrics.functional.average_precision(pred_f, outputs, pos_label=1)
            self.log('test_avg_precision', test_avg_precision, on_epoch=True)

        if "iou" in self.args.metric_log:
            t = torch.tensor(self.args.iou_threshold)
            out_int = outputs.int()
            res_int = (pred_f>t).int()
            test_iou = pl.metrics.functional.classification.iou(res_int, out_int)
            self.log('test_iou', test_iou, on_epoch=True)

        if self.args.developer_mode:
            l = len(inputs)
            for i in range(l):
                img_size = sizes[i]
                rev_size = [img_size[1], img_size[0]]
                f_1 = pred_f.data.cpu()
                f_1_trans = np.array(transforms.Resize(rev_size)(to_pil(f_1[0])))
                f_1_crf = crf_refine(real_inputs[i], f_1_trans)
                img_res = Image.fromarray(f_1_crf)
                new_image = Image.new('RGB',(3*img_size[0], img_size[1]), (250,250,250))
                new_image.paste(Image.fromarray(real_inputs[i]),(0,0))
                new_image.paste(Image.fromarray(real_outputs[i]),(img_size[0],0))
                new_image.paste(img_res,(img_size[0]*2,0))

                new_image.save(os.path.join(self.args.gdd_results_root, self.args.log_name,
                                                        "Test_Epoch: " + str(self.args.iter) +"_Pred.png"))
                self.args.iter = self.args.iter + 1 

        return {'test_loss': loss, 'test_acc': self.test_acc, 'input_size': input_size}


    
    def test_epoch_end(self, test_step_outputs):  # args are defined as part of pl API
        pass

    def infer(self, imag_path):
        img_list = [img_name for img_name in os.listdir(imag_path)]

        for idx, img_name in enumerate(img_list):
            print('predicting for {}: {:>4d} / {}'.format("GDNet", idx + 1, len(img_list)))
            # check_mkdir(os.path.join(self.args.gdd_results_root, '%s_%s' % (exp_name, args['snapshot'])))
            img = Image.open(os.path.join(self.args.infer_path, img_name))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            w, h = img.size
            # img_var = torch.tensor(self.img_transform(img).unsqueeze(0)).cuda(self.args.device_ids[0])
            img_var = torch.tensor(self.img_transform(img).unsqueeze(0))
            f1, f2, f3 = self(img_var)
            f1 = f1.data.squeeze(0).cpu()
            f2 = f2.data.squeeze(0).cpu()
            f3 = f3.data.squeeze(0).cpu()
            f1 = np.array(transforms.Resize((h, w))(to_pil(f1)))
            f2 = np.array(transforms.Resize((h, w))(to_pil(f2)))
            f3 = np.array(transforms.Resize((h, w))(to_pil(f3)))

            if self.args.crf:
                f1 = crf_refine(np.array(img.convert('RGB')), f1)
                f2 = crf_refine(np.array(img.convert('RGB')), f2)
                f3 = crf_refine(np.array(img.convert('RGB')), f3)

            #################################### My Addition ####################################
            image1_size = img.size
            new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
            img_res = Image.fromarray(f3)
            new_image.paste(img,(0,0))
            new_image.paste(img_res,(image1_size[0],0))
            new_image.save(os.path.join(self.args.gdd_results_root, self.args.log_name, img_name[:-4] + "_both" +".png"))
            #####################################################################################

            # if self.args.crf:
            #     f1 = crf_refine(np.array(img.convert('RGB')), f1)
            #     f2 = crf_refine(np.array(img.convert('RGB')), f2)
            #     f3 = crf_refine(np.array(img.convert('RGB')), f3)

            ######################################### Old ############################################
            # Image.fromarray(f1).save(os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']),
            #                                       img_name[:-4] + "_h.png"))
            # Image.fromarray(f2).save(os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']),
            #                                       img_name[:-4] + "_l.png"))
            # Image.fromarray(f3).save(os.path.join(gdd_results_root, '%s_%s' % (exp_name, args['snapshot']),
            #                                       img_name[:-4] + ".png"))


            # Image.fromarray(f1).save(os.path.join(self.args.gdd_results_root, self.args.log_name, img_name[:-4] + "_h.png"))
            # Image.fromarray(f2).save(os.path.join(self.args.gdd_results_root, self.args.log_name, img_name[:-4] + "_l.png"))
            # Image.fromarray(f3).save(os.path.join(self.args.gdd_results_root, self.args.log_name, img_name[:-4] + ".png"))

