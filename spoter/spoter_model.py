import copy
import torch

import torch.nn as nn
from typing import Optional, Union, Callable
from torch import Tensor

import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder
from spoter.attention import AttentionLayer, ProbAttention, FullAttention

isChecked = False


def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])


class FeatureIsolatedTransformer(nn.Transformer):
    def __init__(self, d_model_list: list, nhead_list: list, num_encoder_layers: int = 2, num_decoder_layers: int = 2,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 selected_attn: str = 'full', output_attention: str = True, inner_classifiers: nn.ModuleList = None,
                 patient=0):

        super(FeatureIsolatedTransformer, self).__init__(sum(d_model_list), nhead_list[-1], num_encoder_layers,
                                                         num_decoder_layers, dim_feedforward, dropout, activation)
        del self.encoder
        self.d_model = sum(d_model_list)
        self.d_ff = dim_feedforward
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.activation = activation
        self.selected_attn = selected_attn
        self.output_attention = output_attention
        self.l_hand_encoder = self.get_custom_encoder(d_model_list[0], nhead_list[0])
        self.r_hand_encoder = self.get_custom_encoder(d_model_list[1], nhead_list[1])
        self.body_encoder = self.get_custom_encoder(d_model_list[2], nhead_list[2])
        self.decoder = self.get_custom_decoder(nhead_list[-1], inner_classifiers, patient)
        self._reset_parameters()

    def get_custom_encoder(self, f_d_model: int, nhead: int):
        encoder_layer = TransformerEncoderLayer(f_d_model, nhead, self.d_ff, self.dropout, self.activation)
        if self.selected_attn == 'prob':
            encoder_layer.self_attn = AttentionLayer(
                ProbAttention(output_attention=self.output_attention),
                f_d_model, nhead, mix=False
            )
            print(f'self.selected_attn {self.selected_attn}')

        else:
            encoder_layer.self_attn = AttentionLayer(
                FullAttention(output_attention=self.output_attention),
                f_d_model, nhead, mix=False
            )
            print(f'self.selected_attn {self.selected_attn}')
        encoder_norm = LayerNorm(f_d_model)
        return TransformerEncoder(encoder_layer, self.num_encoder_layers, encoder_norm)

    def get_custom_decoder(self, nhead, inner_classifier, patient):
        decoder_layer = DecoderLayer(self.d_model, nhead, self.d_ff)
        decoder_norm = LayerNorm(self.d_model)
        return Decoder(decoder_layer, self.num_decoder_layers, norm=decoder_norm,
                       patient=patient, inner_classifier=inner_classifier, )

    def checker(self, full_src, tgt, is_batched):
        if not self.batch_first and full_src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and full_src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        if full_src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

    def forward(self, src: list, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                src_is_causal: Optional[bool] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False, training=True) -> Tensor:

        full_src = torch.cat(src, dim=-1)
        self.checker(full_src, tgt, full_src.dim() == 3)
        # print(f"attention in shape: {src[0].shape}")
        l_hand_memory = self.l_hand_encoder(src[0], mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        r_hand_memory = self.r_hand_encoder(src[1], mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        body_memory = self.body_encoder(src[2], mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        full_memory = torch.cat((l_hand_memory, r_hand_memory, body_memory), -1)

        output = self.decoder(tgt, full_memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask, training=training)
        return output


class Decoder(nn.TransformerDecoder):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None, patient=0, inner_classifier=None):
        super(Decoder, self).__init__(decoder_layer, num_layers, norm)
        self.patient = patient
        self.inner_classifier = inner_classifier

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False, training=True) -> Tensor:

        output = tgt

        if training or self.patient == 0:
            for i, mod in enumerate(self.layers):
                output = mod(output, memory, tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask)
                mod_output = output
                if self.norm is not None:
                    mod_output = self.norm(mod_output)
                _ = self.inner_classifier[i](mod_output).squeeze()
        else:
            patient_counter = 0
            patient_result = None
            calculated_layer_num = 0
            for i, mod in enumerate(self.layers):
                calculated_layer_num += 1
                output = mod(output, memory, tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask)

                mod_output = output
                if self.norm is not None:
                    mod_output = self.norm(mod_output)
                classifier_out = self.inner_classifier[i](mod_output).squeeze()
                labels = classifier_out.detach().argmax(dim=1)

                if patient_result is not None:
                    patient_labels = patient_result.detach().argmax(dim=1)
                if (patient_result is not None) and torch.all(labels.eq(patient_labels)):
                    patient_counter += 1
                else:
                    patient_counter = 0

                patient_result = classifier_out
                if patient_counter == self.patient:
                    break

        if self.norm is not None:
            output = self.norm(output)

        return output


class DecoderLayer(nn.TransformerDecoderLayer):
    """
    Edited TransformerDecoderLayer implementation omitting the redundant self-attention operation as opposed to the
    standard implementation.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu):
        super(DecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        # Change self.multihead_attn to use Pro-sparse attention

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None, tgt_is_causal: Optional[bool] = False,
                memory_is_causal: Optional[bool] = False) -> torch.Tensor:
        global isChecked
        if not isChecked:
            print('Using custom DecoderLayer')
            isChecked = True

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

    def __init__(self, num_classes, num_hid=108, attn_type='prob', num_encoder_layers=6,
                 num_decoder_layers=6):
        super(SPOTER, self).__init__()
        print(f"The used pytorch version: {torch.__version__}")
        # self.feature_extractor = FeatureExtractor(num_hid=108, kernel_size=7)
        self.l_hand_encoding = nn.Parameter(self.get_encoding_table(d_model=42))
        self.r_hand_encoding = nn.Parameter(self.get_encoding_table(d_model=42))
        self.body_encoding = nn.Parameter(self.get_encoding_table(d_model=24))

        self.transformer = FeatureIsolatedTransformer(
            [42, 42, 24], [3, 3, 2, 9], selected_attn=attn_type, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, inner_classifiers=nn.ModuleList([
                nn.Linear(num_hid, num_classes) for _ in range(num_decoder_layers)
            ]), patient=1
        )
        self.class_query = nn.Parameter(torch.rand(1, 1, num_hid))
        self.linear_class = nn.Linear(num_hid, num_classes)

        # custom_decoder_layer = DecoderLayer(self.transformer.d_model, self.transformer.nhead)
        # self.transformer.decoder.layers = _get_clones(custom_decoder_layer, self.transformer.decoder.num_layers)

    def forward(self, l_hand, r_hand, body, training):
        batch_size = l_hand.size(0)
        # (batch_size, seq_len, respected_feature_size, coordinates): (24, 204, 54, 2)
        # -> (batch_size, seq_len, feature_size):  (24, 204, 108)
        new_l_hand = l_hand.view(l_hand.size(0), l_hand.size(1), l_hand.size(2) * l_hand.size(3))
        new_r_hand = r_hand.view(r_hand.size(0), r_hand.size(1), r_hand.size(2) * r_hand.size(3))
        body = body.view(body.size(0), body.size(1), body.size(2) * body.size(3))

        # (batch_size, seq_len, feature_size) : (24, 204, 108)
        # -> (seq_len, batch_size, feature_size): (204, 24, 108)
        new_l_hand = new_l_hand.permute(1, 0, 2).type(dtype=torch.float32)
        new_r_hand = new_r_hand.permute(1, 0, 2).type(dtype=torch.float32)
        new_body = body.permute(1, 0, 2).type(dtype=torch.float32)

        # feature_map = self.feature_extractor(new_inputs)
        # transformer_in = feature_map + self.pos_embedding
        l_hand_in = new_l_hand + self.l_hand_encoding  # Shape remains the same
        r_hand_in = new_r_hand + self.r_hand_encoding  # Shape remains the same
        body_in = new_body + self.body_encoding  # Shape remains the same

        # (seq_len, batch_size, feature_size) -> (batch_size, 1, feature_size): (24, 1, 108)
        transformer_output = self.transformer(
            [l_hand_in, r_hand_in, body_in], self.class_query.repeat(1, batch_size, 1), training=training
        ).transpose(0, 1)

        # (batch_size, 1, feature_size) -> (batch_size, num_class): (24, 100)
        out = self.linear_class(transformer_output).squeeze()
        return out

    @staticmethod
    def get_encoding_table(d_model=108, seq_len=204):
        torch.manual_seed(42)
        tensor_shape = (seq_len, d_model)
        frame_pos = torch.rand(tensor_shape)
        for i in range(tensor_shape[0]):
            for j in range(1, tensor_shape[1]):
                frame_pos[i, j] = frame_pos[i, j - 1]
        frame_pos = frame_pos.unsqueeze(1)  # (seq_len, 1, feature_size): (204, 1, 108)
        return frame_pos
