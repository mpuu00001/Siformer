# spotter work

import copy
import torch

import torch.nn as nn
from typing import Optional

import torch.nn.functional as F


def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])


class SPOTERTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Edited TransformerDecoderLayer implementation omitting the redundant self-attention operation as opposed to the
    standard implementation.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super(SPOTERTransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        # del self.self_attn

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None, tgt_is_causal: Optional[bool] = False,
                memory_is_causal: Optional[bool] = False) -> torch.Tensor:

        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class FeatureExtractor(nn.Module):
    def __init__(self, num_hid=108, kernel_size=9):
        super(FeatureExtractor, self).__init__()
        # self.frame_wise_pos = nn.Parameter(frame_embedding(max_len, num_hid))
        self.num_hid = num_hid
        self.conv1 = nn.Conv1d(num_hid, num_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=1)
        self.conv2 = nn.Conv1d(num_hid, num_hid, kernel_size=kernel_size-2, padding=(kernel_size - 3) // 2, stride=1)
        self.conv3 = nn.Conv1d(num_hid, num_hid, kernel_size=kernel_size-4, padding=(kernel_size - 5) // 2, stride=1)
        self.conv4 = nn.Conv1d(num_hid, num_hid, kernel_size=kernel_size-6, padding=(kernel_size - 7) // 2, stride=1)

        self.bn1 = nn.BatchNorm1d(num_hid, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(num_hid, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.bn3 = nn.BatchNorm1d(num_hid, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.bn4 = nn.BatchNorm1d(num_hid, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
          # Turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len) for CNN
        x = x.permute(1, 2, 0)    # (16, 108, 204)
        x = x.to(torch.float32)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = out.permute(2, 0, 1)      # (204,16,108)
        out = out * torch.sqrt(torch.tensor(self.num_hid, dtype=torch.float32))
        # x = x + self.frame_wise_pos
        return out


class SPOTER(nn.Module):
    """
    Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
    of skeletal data.
    """

    def __init__(self, num_classes, num_hid=108, batch_size=16):
        super(SPOTER, self).__init__()
        # self.feature_extractor = FeatureExtractor(num_hid=108, kernel_size=7)
        self.pos_embedding = nn.Parameter(self.get_encoding_table())
        self.class_query = nn.Parameter(torch.rand(1, 1, num_hid))
        self.transformer = nn.Transformer(num_hid, 9, 6, 6)
        self.linear_class = nn.Linear(num_hid, num_classes)
        custom_decoder_layer = SPOTERTransformerDecoderLayer(self.transformer.d_model, self.transformer.nhead, 2048, 0.1, "relu")
        self.transformer.decoder.layers = _get_clones(custom_decoder_layer, self.transformer.decoder.num_layers)

    @staticmethod
    def get_encoding_table(num_hid=108, num_frame=204):
        torch.manual_seed(42)
        tensor_shape = (num_frame, num_hid)
        frame_pos = torch.rand(tensor_shape)
        for i in range(tensor_shape[0]):
            for j in range(1, tensor_shape[1]):
                frame_pos[i, j] = frame_pos[i, j - 1]
        frame_pos = frame_pos.unsqueeze(1)

        return frame_pos

    def forward(self, inputs):
        new_inputs = inputs.view(inputs.size(0), inputs.size(1), inputs.size(2)*inputs.size(3)) # (204,16,108) seq_len, batch_size ,feature_size
        new_inputs = new_inputs.permute(1, 0, 2).type(dtype=torch.float32)
        # feature_map = self.feature_extractor(new_inputs)
        # transformer_in = feature_map + self.pos_embedding
        transformer_in = new_inputs + self.pos_embedding
        transformer_output = self.transformer(transformer_in, self.class_query.repeat(1, new_inputs.size(1), 1)).transpose(0, 1)
        out = self.linear_class(transformer_output).squeeze()
        return out
