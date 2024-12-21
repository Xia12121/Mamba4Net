import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from transformers.utils.dummy_pt_objects import PreTrainedModel
from config import cfg

# 这里假设已经定义好了 MambaModel 类
# from somewhere import MambaModel

class Mamba_pipeline(nn.Module):
    '''
    Pipeline for viewport prediction.
    Pipeline for Mamba distillation.
    '''
    def __init__(self,
                 plm: PreTrainedModel,
                 mamba_plm,  # 这里用你的 MambaModel 或者对应类 (学生模型)
                 loss_func = None,
                 fut_window = None,
                 device = 'cuda',
                 embed_size = 1024,
                 frequency = 5,
                 using_multimodal = False,
                 dataset = None,
                 alpha_kl = 1.0  # KL损失在总loss中的权重，可自行调节
                ):
        """
        :param plm: the pretrained LLM (teacher model)
        :param mamba_plm: the MambaModel (student model)
        :param embed_size: the embed size of LLM
        :param frequency: the frequency of dataset
        :param fut_window: future (prediction) window
        :param dataset: the dataset
        :param using_multimodal: adding multimodal image features (True/False)
        :param device: cuda or cpu
        :param alpha_kl: KL损失的权重
        """
        super().__init__()
        self.plm = plm
        self.mamba_plm = mamba_plm  # 学生模型
        self.using_multimodal = using_multimodal
        self.dataset = dataset
        self.device = device
        self.frequency = frequency
        self.embed_size = embed_size
        self.fut_window_length = fut_window
        self.alpha_kl = alpha_kl

        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 256, 3),
            nn.LeakyReLU(),
            nn.Flatten()
        ).to(device)

        self.embed_vp = nn.Linear(256, self.embed_size).to(device)
        self.embed_multimodal = nn.Linear(768, embed_size).to(device)  # 768 = ViT output feature size
        self.embed_ln = nn.LayerNorm(self.embed_size).to(device)

        # 用来在 get_multimodal_information 中缓存读取好的特征
        self.loaded_tensor_cache = {}

        # 根据项目需要把除 plm 外的关键层加进来
        self.modules_except_plm = nn.ModuleList([
            self.embed_vp,
            self.embed_multimodal,
            self.embed_ln,
            self.conv1d,
            self.plm.networking_head
        ])

        if loss_func is None:
            loss_func = nn.MSELoss()
        self.loss_fct = loss_func  # 用于教师或学生的端到端监督误差

        # 用于逐层隐藏状态的 KL 散度损失
        # 在 PyTorch 中常见的 KLDivLoss 需要输入 log_prob 和 prob
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

        self.fut_window = fut_window
    
    def forward(self, batch, future, video_user_position, teacher_forcing=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param batch: history viewport trajectory
        :param future: future viewport trajectory
        :param video_user_position: details information for current trajectory
        :return: (teacher_loss, student_loss)
            teacher_loss: 教师模型的端到端监督损失
            student_loss: 学生模型端到端监督损失 + KL 散度损失
        """
        # 1. 准备输入特征(embedding)
        x = self._prepare_sequence_embeddings(batch, future, video_user_position, teacher_forcing)

        # 2. 同时前向传播：
        #   - 教师模型 (plm) 输出 teacher_out
        #   - 学生模型 (mamba_plm) 输出 student_out
        # 注意这里需要设置 return_dict=True, output_hidden_states=True, 
        #  这样才能拿到 hidden_states；如果 transformer 版本不同可自行适配
        teacher_out = self.plm(
            inputs_embeds=x,
            attention_mask=torch.ones(x.shape[0], x.shape[1], dtype=torch.long, device=self.device),
            output_hidden_states=True,  # 为了拿到隐藏层
            return_dict=True
        )

        student_out = self.mamba_plm(
            inputs_embeds=x,
            attention_mask=torch.ones(x.shape[0], x.shape[1], dtype=torch.long, device=self.device),
            output_hidden_states=True,
            return_dict=True
        )

        # 3. 分别计算教师模型和学生模型的损失

        # 3.1 教师模型端到端损失(teacher_loss)
        # 这里一般就是对 teacher_out.logits 和 ground truth 做监督
        gt = future.to(self.device)
        teacher_loss = self.loss_fct(teacher_out.logits, gt)

        # 3.2 学生模型端到端损失(student_e2e_loss) + KL散度损失
        student_e2e_loss = self.loss_fct(student_out.logits, gt)

        # 计算逐层 KL 散度
        # 假设 teacher_out.hidden_states 和 student_out.hidden_states 的长度相同
        # 如果层数不一致，需要自己做好层的对齐和映射
        layer_kl_loss = 0.0
        teacher_hidden_states = teacher_out.hidden_states
        student_hidden_states = student_out.hidden_states
        
        # 通常 hidden_states[0] 是 embedding 输出, hidden_states[-1] 是最后一层输出
        # 大多数情况下会对中间的几层做蒸馏(可以自己控制哪些层做对齐)
        for t_hid, s_hid in zip(teacher_hidden_states, student_hidden_states):
            # KLDivLoss 需要 log_prob vs prob
            # 所以对学生做 log_softmax，对教师做 softmax
            s_log_prob = F.log_softmax(s_hid, dim=-1)
            t_prob = F.softmax(t_hid, dim=-1)

            layer_kl_loss += self.kl_div(s_log_prob, t_prob)

        # 最终学生的损失
        student_loss = student_e2e_loss + self.alpha_kl * layer_kl_loss

        # 4. 返回 (teacher_loss, student_loss)
        # 在训练逻辑里可以分别对 teacher_loss.backward() 和 student_loss.backward()，
        # 或者根据需求只回传学生梯度。
        return teacher_loss, student_loss

    def _prepare_sequence_embeddings(self, batch, future, video_user_position, teacher_forcing):
        """
        按照原始 pipeline 的思路，得到 [batch_size, seq_len, embed_dim] 的输入。
        如果是 teacher_forcing=True，则把 future 拼到序列后面；
        如果是 auto-regressive，则只拿 history。
        """
        if teacher_forcing:
            # teaching_forcing 的情况
            x = torch.cat((batch, future), dim=1)
        else:
            # auto_regressive 的情况
            x = batch  # 后面可根据实际需要写自回归逻辑

        seq_len = x.shape[1]
        batch_embeddings = []
        for i in range(seq_len):
            # conv1d 输入需要 [batch_size, channel, length] 形式
            # 此时 x[:, i, :] 大小是 [batch_size, embed_dim]，故 reshape 成 [batch_size, 1, embed_dim]
            conv_inp = x[:, i, :].unsqueeze(1)
            conv_out = self.conv1d(conv_inp)  # [batch_size, 256]
            embed_vp_out = self.embed_vp(conv_out)  # [batch_size, embed_size]
            batch_embeddings.append(embed_vp_out.unsqueeze(1))
        
        # 拼接回 [batch_size, seq_len, embed_size]
        x = torch.cat(batch_embeddings, dim=1)

        # 如果需要多模态特征，则拼接
        if self.using_multimodal:
            mapped_tensor = self.get_multimodal_information(video_user_position)
            # mapped_tensor 大小是 [batch_size, 1, embed_size]
            x = torch.cat([mapped_tensor, x], dim=1)

        # 过 LayerNorm
        x = self.embed_ln(x)
        return x

    def get_multimodal_information(self, video_user_position):
        """
        从磁盘缓存中读取对于指定 (video_index, position_index) 的图像特征，
        并映射到 embed_size 维度。与原 pipeline 相同。
        """
        video_index = video_user_position[0].item()
        position_index = video_user_position[2].item()
        image_index = (position_index - 1) * (cfg.video_frame[self.dataset][video_index-1] // self.frequency)

        if image_index % 100 == 0:
            cache_key = f'{video_index}_{image_index // 100}'
        else:
            cache_key = f'{video_index}_{(image_index // 100) + 1}'

        if cache_key in self.loaded_tensor_cache:
            loaded_tensor_dict = self.loaded_tensor_cache[cache_key]
        else:
            if image_index % 100 == 0:
                loaded_tensor_dict = torch.load(
                    os.path.join(cfg.dataset_image_features[self.dataset],
                                 f'video{video_index}_images/feature_dict{(image_index // 100)}.pth')
                )
            else:
                loaded_tensor_dict = torch.load(
                    os.path.join(cfg.dataset_image_features[self.dataset],
                                 f'video{video_index}_images/feature_dict{(image_index // 100) + 1}.pth')
                )
            self.loaded_tensor_cache[cache_key] = loaded_tensor_dict

        load_tensor = loaded_tensor_dict[f'{image_index}'].to(self.device)
        mapped_tensor = self.embed_multimodal(load_tensor)  # [batch_size, embed_size]，假如是1条则 [1, embed_size]
        mapped_tensor = mapped_tensor.unsqueeze(1)  # [batch_size, 1, embed_size]
        return mapped_tensor

    # 如果需要保留原 auto_regressive / teaching_forcing / inference 等逻辑，也可以在此处自行添加。
    # 但要注意，你现在的 forward 做的是【一次同时跑 teacher & student】，并分别返回两种 loss。
    # 具体训练调用时，可根据需要拆分或合并。


    def auto_regressive(self, x, future, video_user_position) -> torch.Tensor:
        """
        auto-regressive generation
        
        :return: the loss value for training
        """
        seq_len = x.shape[1]
        batch_embeddings = []
        for i in range(seq_len):
            batch_embeddings.append(self.embed_vp(self.conv1d(x[:, i, :]).view(1,256)).unsqueeze(1))
        x = torch.cat(batch_embeddings, dim=1)

        if self.using_multimodal:  # we make using multimodal image features as an option, as not all datasets provide video information.
            mapped_tensor = self.get_multimodal_information(video_user_position)
            x = torch.cat([mapped_tensor, x], dim=1)

        x = self.embed_ln(x)

        outputlist = []
        for _ in range(self.fut_window_length):
            outputs = self.plm(inputs_embeds=x, attention_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.long, device=self.device))
            outputlist.append(outputs.logits)
            x = torch.cat((x, self.embed_vp(self.conv1d(outputs.logits)).unsqueeze(1)), dim=1)

        pred = torch.cat(outputlist, dim=1)
        return pred
    
    def teaching_forcing(self, x, future, video_user_position) -> torch.Tensor:
        """
        teaching-forcing generation

        :param x: history viewport trajectory
        :param future: future viewport trajectory
        :param video_user_position: details information for current trajectory
        :return: the return value by llm
        """

        x = torch.cat((x, future), dim=1)
        seq_len = x.shape[1]
        batch_embeddings = []
        for i in range(seq_len):
            batch_embeddings.append(self.embed_vp(self.conv1d(x[:, i, :]).view(1,256)).unsqueeze(1))
        x = torch.cat(batch_embeddings, dim=1)

        if self.using_multimodal:
            mapped_tensor = self.get_multimodal_information(video_user_position)
            x = torch.cat([mapped_tensor, x], dim=1)
        
        x = self.embed_ln(x)

        outputs = self.plm(inputs_embeds=x, attention_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.long, device=self.device), teacher_forcing=True)
        return outputs.logits
    
    def inference(self, batch, future, video_user_info) -> torch.Tensor:
        """
        Inference function. Use it for testing.
        """
        pred = self.auto_regressive(batch, future, video_user_info)
        gt = future.to(pred.device)
        return pred, gt
