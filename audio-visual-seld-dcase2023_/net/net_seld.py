# Copyright 2023 Sony Group Corporation.

import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from net.net_util import interpolate
from timm.models.layers import to_2tuple, DropPath, trunc_normal_
from vision_transformer import VisionTransformer, Block


def create_net_seld(args):
    with open(args.feature_config, 'r') as f:
        feature_config = json.load(f)
    if args.net == 'crnn':
        Net = AudioVisualCRNN(class_num=args.class_num,
                              in_channels=feature_config[args.feature]["ch"])
    elif args.net == 'vis_transformer':
        Net = AudioVisualCRNN(class_num=args.class_num,
                              in_channels=feature_config[args.feature]["ch"],
                              audio_blk="vit")
    return Net


class AudioVisualCRNN(nn.Module):
    def __init__(self, class_num, in_channels, interp_ratio=16, audio_blk="cnn"):
        super().__init__()
        # print(f"audio_blk: {audio_blk}")
        self.class_num = class_num
        self.interp_ratio = interp_ratio

        # Audio
        aud_embed_size = 64
        if audio_blk == "cnn":
            self.audio_encoder = CNN3(in_channels=in_channels, out_channels=aud_embed_size)
        elif audio_blk == "vit":
            self.audio_encoder = Vit(in_channels=in_channels, out_channels=64)

        # Visual
        vis_embed_size = 64
        vis_in_size = 2 * 6 * 37
        project_vis_embed_fc1 = nn.Linear(vis_in_size, vis_embed_size)
        project_vis_embed_fc2 = nn.Linear(vis_embed_size, vis_embed_size)
        self.vision_encoder = nn.Sequential(project_vis_embed_fc1,
                                            project_vis_embed_fc2)

        # Audio-Visual
        in_size_gru = aud_embed_size + vis_embed_size
        self.gru = nn.GRU(input_size=in_size_gru, hidden_size=256,
                          num_layers=1, batch_first=True, bidirectional=True)
        self.fc_xyz = nn.Linear(512, 3 * 3 * self.class_num, bias=True)

    def forward(self, x_a, x_v):
        x_a = x_a.transpose(2, 3)
        b_a, c_a, t_a, f_a = x_a.size()  # input: batch_size, mic_channels, time_steps, freq_bins
        b, c, t, f = b_a, c_a, t_a, f_a
        # print('x_a.shape', x_a.shape)
        x_a = self.audio_encoder(x_a)
        # x_a = torch.mean(x_a, dim=3)  # x_a: batch_size, feature_maps, time_steps
        # print('x_a.shape', x_a.shape)

        x_v = x_v.view(x_v.size(0), -1)
        x_v = self.vision_encoder(x_v)
        x_v = torch.unsqueeze(x_v, dim=-1).repeat(1, 1, 8)  # repeat for time_steps
        # print('x_v.shape', x_v.shape)
        
        # create a all 0 tensor like x_v if no vision input
        x_v_0 = torch.zeros_like(x_v)
        
        x = torch.cat((x_a, x_v), 1)

        x = x.transpose(1, 2)  # x: batch_size, time_steps, feature_maps
        self.gru.flatten_parameters()
        (x, _) = self.gru(x)

        x = self.fc_xyz(x)  # event_output: batch_size, time_steps, 3 * 3 * class_num
        x = interpolate(x, self.interp_ratio)
        x = x.transpose(1, 2)
        x = x.view(-1, 3, 3, self.class_num, t)

        return x


class CNN3(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.conv3 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        print('x.shape', x.shape)
        x = F.relu_(self.bn1(self.conv1(x)))
        print('after conv1, x.shape', x.shape)
        x = F.max_pool2d(x, kernel_size=(4, 4))
        print('after maxpool1, x.shape', x.shape)
        x = F.relu_(self.bn2(self.conv2(x)))
        print('after conv2, x.shape', x.shape)
        x = F.max_pool2d(x, kernel_size=(2, 4))
        print('after maxpool2, x.shape', x.shape)
        x = F.relu_(self.bn3(self.conv3(x)))
        print('after conv3, x.shape', x.shape)
        x = F.max_pool2d(x, kernel_size=(2, 2))
        print('after maxpool3, x.shape', x.shape)
        x = torch.mean(x, dim=3)
        return x


class Vit(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=1,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.conv2 = Block(dim=128, num_heads=4)
        self.conv3 = Block(dim=128, num_heads=4)
        self.bn1 = nn.BatchNorm1d(out_channels*2)

    def forward(self, x):
        x = F.relu_(self.bn1(self.conv1(x).squeeze())).transpose(1,2)  # x: batch_size, 1, time_steps, freq_bins 
        x = F.max_pool2d(x, kernel_size=(4, 1)) # x: batch_size, time_steps/4, freq_bins/4 
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=(4, 1)) # x: batch_size, time_steps/8, freq_bins/16 
        x = self.conv3(x)
        x = F.max_pool2d(x, kernel_size=(2, 2)) # x: batch_size, 1, time_steps/16, freq_bins/64 
        return x.transpose(1, 2) # x: batch_size, freq_bins/64, time_steps/16, 1