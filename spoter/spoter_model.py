
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


class SPOTER(nn.Module):
    """
    Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
    of skeletal data.
    """

    def __init__(self, num_classes, num_seq_elements=108):
        super().__init__()
        self.frame_wise_pos = nn.Parameter(frame_wise_embedding_matrix())
        self.class_query = nn.Parameter(torch.rand(1, num_seq_elements))
        self.transformer = nn.Transformer(num_seq_elements, 9, 6, 6)
        self.linear_class = nn.Linear(num_seq_elements, num_classes)

        # Deactivate the initial attention decoder mechanism
        custom_decoder_layer = SPOTERTransformerDecoderLayer(self.transformer.d_model, self.transformer.nhead, 2048,
                                                             0.1, "relu")
        self.transformer.decoder.layers = _get_clones(custom_decoder_layer, self.transformer.decoder.num_layers)

    def forward(self, inputs):
        new_inputs = torch.unsqueeze(inputs.flatten(start_dim=1), 1).float()
        new_inputs = self.transformer(self.frame_wise_pos + new_inputs, self.class_query.unsqueeze(0)).transpose(0, 1)
        out = self.linear_class(new_inputs)
        return out


def frame_wise_embedding_matrix(num_frame=204, num_seq_elements=108):

    # Set a seed for reproducibility
    torch.manual_seed(42)

    # Define the shape of the tensor
    tensor_shape = (num_frame, num_seq_elements)

    # Initialize the tensor with random values between [0, 1]
    frame_pos = torch.rand(tensor_shape)

    # Ensure that values within the same inner matrix are the same
    for i in range(tensor_shape[0]):
        for j in range(1, tensor_shape[1]):
            frame_pos[i, j] = frame_pos[i, j - 1]

    res = frame_pos.unsqueeze(1)

    return res


if __name__ == "__main__":
    pass
