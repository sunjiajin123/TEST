"""
AIO -- All Trains in One
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal

from trains.singleTask import *
from trains.multiTask import *

__all__ = ['ATIO']

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            # single-task
            'tfn': TFN,
            'lmf': LMF,
            'mfn': MFN,
            'ef_lstm': EF_LSTM,
            'lf_dnn': LF_DNN,
            'graph_mfn': Graph_MFN,
            'mult': MULT,
            # 'bert_mag':BERT_MAG,
            'misa': MISA,
            # multi-task
            'mtfn': MTFN,
            'mlmf': MLMF,
            'mlf_dnn': MLF_DNN,
            'self_mm': SELF_MM,
            'test':TEST,
            'test1':TEST1,
            'rcm':RCM,
            'rcm_1': RCM_1,
            'rcm_2': RCM_2,
            'rcm_3': RCM_3,
            'rcm_4': RCM_4,
            'rcm_5': RCM_5

        }
    
    def getTrain(self, args):
        return self.TRAIN_MAP[args.modelName.lower()](args)
