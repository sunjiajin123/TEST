"""
paper1: Benchmarking Multimodal Sentiment Analysis
paper2: Recognizing Emotions in Video Using Multimodal DNN Feature Fusion
From: https://github.com/rhoposit/MultimodalDNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.subNets.FeatureNets import SubNet, TextSubNet
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
__all__ = ['EF_LSTM']


class TEST1(nn.Module):
    """
    early fusion using lstm
    """

    def __init__(self, args):
        super(TEST1, self).__init__()
        text_in, audio_in, video_in = args.feature_dims
        in_size = text_in + audio_in + video_in
        input_len = args.seq_lens
        hidden_size = args.hidden_dims
        num_layers = args.num_layers
        dropout = args.dropout
        output_dim = args.num_classes if args.train_mode == "classification" else 1
        self.norm = nn.BatchNorm1d(input_len)
        self.lstm = nn.LSTM(text_in, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=False,
                            batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_dim)

        self.text_subnet = TextSubNet(audio_in,128 , 64, dropout=0.2)            #新加入
        self.mylinear=nn.Linear(64,3)                                       #新加入
    def forward(self, text_x, audio_x, video_x):
        # early fusion (default: seq_l == seq_a == seq_v)
        audio, audio_lengths = audio_x
        video, video_lengths = video_x
        packed_sequence = pack_padded_sequence(audio, audio_lengths, batch_first=True, enforce_sorted=False)
        x = self.text_subnet(packed_sequence)
        x = self.mylinear(x)


        # _, final_states = self.lstm(x)
        # x = self.dropout(final_states[0][-1].squeeze())
        # x = F.relu(self.linear(x), inplace=True)
        # x = self.dropout(x)
        # output = self.out(x)
        res = {
            'M': x
        }
        return res


class EF_CNN(nn.Module):
    """
    early fusion using cnn
    """

    def __init__(self, args):
        super(EF_CNN, self).__init__()

    def forward(self, text_x, audio_x, video_x):
        pass