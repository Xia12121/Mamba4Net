import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from mamba_ssm import Mamba2

class MambaLlamaNetworkingHeadModel(nn.Module):
    _tied_weights_keys = ["networking_head.weight"]

    def __init__(self, config):
        """
        config 里需要包含:
          - hidden_size: 每层输入/输出的特征维度 d_model
          - num_hidden_layers: 层数，需与 Llama 对齐
          - d_state, d_conv, expand 等 Mamba2 的超参数 (也可固定写死)
        """
        super().__init__()

        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size

        # 下面参数根据需要从 config 中取，或写死
        d_state = getattr(config, "d_state", 64)
        d_conv = getattr(config, "d_conv", 4)
        expand = getattr(config, "expand", 2)

        # 构建多层 Mamba2 block：每层输入输出都是 (B, L, hidden_size)
        self.mamba_blocks = nn.ModuleList([
            Mamba2(
                d_model=self.hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            for _ in range(self.num_hidden_layers)
        ])

        # networking_head（可自定义：例如线性层、MLP等）
        self.networking_head = None  # 外部通过 set_networking_head() 注入

    def get_networking_head(self):
        return self.networking_head

    def set_networking_head(self, networking_head):
        self.networking_head = networking_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_ids_len: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        teacher_forcing: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        与 LlamaNetworkingHeadModel 类似，但换成多层 Mamba2 来堆叠。
        逐层收集 hidden_states 以便层对层的知识蒸馏。
        """
        if inputs_embeds is None:
            raise ValueError("MambaNetworkingHeadModel requires `inputs_embeds` as input.")

        # ------------- Step 1. 逐层前向传播 -------------
        hidden_states = inputs_embeds  # [batch_size, seq_len, hidden_size]
        all_hidden_states = [] if output_hidden_states else None

        for layer_idx, block in enumerate(self.mamba_blocks):
            # 可以在这里对 attention_mask 做一些处理，但通常 Mamba2 不需要多头注意力 mask
            hidden_states = block(hidden_states)  # [batch_size, seq_len, hidden_size]

            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        # ------------- Step 2. 调用 networking_head -------------
        if self.networking_head is not None:
            if teacher_forcing and hasattr(self.networking_head, "teacher_forcing"):
                prediction = self.networking_head.teacher_forcing(hidden_states)
            else:
                prediction = self.networking_head(hidden_states)
        else:
            # 没有设置 networking_head 时，默认输出最后一层 hidden_states
            prediction = hidden_states

        # ------------- Step 3. 打包输出 -------------
        if not return_dict:
            # 返回 tuple，与 Transformers 兼容
            # 例如: (prediction, all_hidden_states, ...)
            output_tuple = (prediction,)
            if output_hidden_states:
                output_tuple += (all_hidden_states,)
            return output_tuple
        else:
            # 返回 huggingface 风格的 CausalLMOutputWithPast
            # 注意: Mamba2 暂无 past_key_values / attentions 概念，可传 None
            return CausalLMOutputWithPast(
                loss=None,
                logits=prediction,
                past_key_values=None,
                hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
                attentions=None
            )
