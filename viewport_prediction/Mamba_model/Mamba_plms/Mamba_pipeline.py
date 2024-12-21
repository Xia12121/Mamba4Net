import os
import torch
import torch.nn as nn
from typing import *
from transformers.utils.dummy_pt_objects import PreTrainedModel
from config import cfg

class Mamba_pipeline(nn.Module):
    '''
    Pipeline for viewport prediction.
    Pipeline for Mamba distillation.
    '''
    def __init__(self,
                plm: PreTrainedModel,
                mamba_plm: MambaModel,
                loss_func = None,
                fut_window = None,
                device = 'cuda',
                embed_size = 1024,
                frequency = 5,
                using_multimodal = False,
                dataset = None
                ):
        """
        :param plm: the pretrained llm
        :param mamba_plm: the mamba llm
        :param embed_size: the embed size of llm
        :param frequency: the frequency of dataset
        :param fut_window: future (prediction) window
        :param dataset: the dataset
        :param using_multimodal: adding multimodal image features (True/False)
        :param device: cuda or cpu
        """
        super().__init__()
        self.plm = plm
        self.mamba_plm = mamba_plm # MambaModel (distillation model)
        self.using_multimodal = using_multimodal
        self.dataset = dataset
        self.device = device
        self.frequency = frequency
        self.embed_size = embed_size
        self.fut_window_length = fut_window

        self.conv1d = nn.Sequential(nn.Conv1d(1, 256, 3), nn.LeakyReLU(), nn.Flatten()).to(device)
        self.embed_vp = nn.Linear(256, self.embed_size).to(device)
        self.embed_multimodal = nn.Linear(768, embed_size).to(device)  # 768 = ViT output feature size
        self.embed_ln = nn.LayerNorm(self.embed_size).to(device)

        self.loaded_tensor_cache = {}
        self.modules_except_plm = nn.ModuleList([  # used to save and load modules except plm
            self.embed_vp, self.embed_multimodal, self.embed_ln, self.conv1d, self.plm.networking_head
        ])

        if loss_func is None:
            loss_func = nn.MSELoss()
        self.loss_fct = loss_func
        self.fut_window = fut_window