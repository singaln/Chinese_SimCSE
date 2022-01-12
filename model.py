# coding=utf-8
# @Time:2021/6/239:34
# @author: SinGaln

"""SimCSE复现"""
import torch
from transformers import BertModel, BertPreTrainedModel

class SimCSE(BertPreTrainedModel):
    def __init__(self, config, args):
        super(SimCSE, self).__init__(config)
        self.args = args
        self.bert = BertModel(config=config)

    def forward(self, input_ids, attention_mask, token_type_ids, encoder_type="first-last-avg"):
        """
        :param input_ids:输入的文本token
        :param attention_mask: 输入文本的mask
        :param token_type_ids: 输入文本字符的类别(no.1 or no.2)
        :return:
        """
        outputs = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        # print(outputs)
        sequence_outputs, pooled_outputs, hidden_states = outputs[0], outputs[1], outputs[2]

        if encoder_type == "first-last-avg":
            first_layer_output = hidden_states[1] # 第0层为embedding layer shape:[batch_size, seq_len, hidden_size]
            last_layer_output = hidden_states[-1] # shape:[batch_size, seq_len, hidden_size]
            seq_length = first_layer_output.size(1)
            # print("first_layer_output", first_layer_output, first_layer_output.shape)
            first_layer_avg = torch.avg_pool1d(first_layer_output.transpose(1,2), kernel_size=seq_length).squeeze(-1) # [batch_size, hidden_size]
            last_layer_avg = torch.avg_pool1d(last_layer_output.transpose(1,2), kernel_size=seq_length).squeeze(-1)
            output = torch.avg_pool1d(torch.cat([first_layer_avg.unsqueeze(1), last_layer_avg.unsqueeze(1)], dim=1).transpose(1,2), kernel_size=2).squeeze(-1)
            return output

        if encoder_type == "last-avg":
            seq_length = sequence_outputs.size(1)
            output = torch.avg_pool1d(sequence_outputs.transpose(1,2), kernel_size=seq_length).squeeze(-1)
            return output

        if encoder_type == "cls":
            cls = sequence_outputs[:,0]
            return cls

        if encoder_type == "pooler":
            return pooled_outputs
