"""
paper: Tensor Fusion Network for Multimodal Sentiment Analysis
From: https://github.com/A2Zadeh/TensorFusionNetwork
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal
from  tqdm import  tqdm
from models.subNets.FeatureNets import SubNet, TextSubNet

class TEST(nn.Module):
    def __init__(self, args):
        super(TEST, self).__init__()
        # dimensions are specified in the order of audio, video and text
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims
        self.output_dim = args.num_classes if args.train_mode == "classification" else 1

        self.text_out= args.text_out                                                                     #64
        self.post_fusion_dim = args.post_fusion_dim                                                     #128

        self.audio_prob, self.video_prob, self.text_prob, self.post_fusion_prob = args.dropouts
        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)                #LSTM  768输入 128隐藏  dropout0.2 通过Liner 64输出
        self.norm = nn.BatchNorm1d(args.seq_lens[0])
        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, self.output_dim)

        # in TFN we are doing a regression with constrained output range: (-3, 3), hence we'll apply sigmoid to output
        # shrink it to (0, 1), and scale\shift it back to range (-3, 3)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
        self.myliner=nn.Linear(self.text_out,self.output_dim)
    def forward(self, text_x, audio_x, video_x):
        text_x=self.norm(text_x)
        text_x=text_x.squeeze(1)
        text_h = self.text_subnet(text_x)
        output=self.myliner(text_h)
        return output













































