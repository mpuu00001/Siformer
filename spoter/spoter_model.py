import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

import copy
from typing import Optional, Union, Callable
from spoter.attention import AttentionLayer, ProbAttention, FullAttention
from spoter.utils import get_sequence_list

is_dec_layer_checked = False
is_enc_checked = False


def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])


class FeatureIsolatedTransformer(nn.Transformer):
    def __init__(self, d_model_list: list, nhead_list: list, num_enc_layers: int, num_dec_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 selected_attn: str = 'prob', output_attention: str = True, inner_classifiers: nn.ModuleList = None,
                 patient=0, use_pyramid_encoder: bool = True, distil: bool = True, enc_layer_list: list = None):

        super(FeatureIsolatedTransformer, self).__init__(sum(d_model_list), nhead_list[-1], num_enc_layers,
                                                         num_dec_layers, dim_feedforward, dropout, activation)
        del self.encoder
        self.d_model = sum(d_model_list)
        self.d_ff = dim_feedforward
        self.dropout = dropout
        self.num_enc_layers = num_enc_layers
        self.enc_layer_list = enc_layer_list
        self.num_dec_layers = num_dec_layers
        self.activation = activation
        self.selected_attn = selected_attn
        self.output_attention = output_attention
        self.use_pyramid_encoder = use_pyramid_encoder
        self.distil = distil
        self.l_hand_encoder = self.get_custom_encoder(d_model_list[0], nhead_list[0])
        self.r_hand_encoder = self.get_custom_encoder(d_model_list[1], nhead_list[1])
        self.body_encoder = self.get_custom_encoder(d_model_list[2], nhead_list[2])
        self.decoder = self.get_custom_decoder(nhead_list[-1], inner_classifiers, patient)
        self._reset_parameters()

    def get_custom_encoder(self, f_d_model: int, nhead: int):
        Attn = ProbAttention if self.selected_attn == 'prob' else FullAttention
        global is_enc_checked
        if not is_enc_checked:
            print(f'self.selected_attn {self.selected_attn}')
            print(f'self.use_pyramid_encoder {self.use_pyramid_encoder}')
            print(f'self.distl {self.distil}')
            is_enc_checked = True

        if not self.use_pyramid_encoder:
            encoder_layer = TransformerEncoderLayer(f_d_model, nhead, self.d_ff, self.dropout, self.activation)
            encoder_layer.self_attn = AttentionLayer(
                Attn(output_attention=self.output_attention),
                f_d_model, nhead, mix=False
            )
            encoder_norm = LayerNorm(f_d_model)
            encoder = TransformerEncoder(encoder_layer, self.num_enc_layers, encoder_norm)
        else:
            e_layers = get_sequence_list(self.num_enc_layers)
            inp_lens = list(range(len(e_layers)))
            encoders = [
                Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                Attn(output_attention=self.output_attention),
                                f_d_model, nhead, mix=False),
                            f_d_model,
                            self.d_ff,
                            dropout=self.dropout,
                            activation=self.activation
                        ) for _ in range(el)
                    ],
                    [
                        ConvLayer(
                            f_d_model
                        ) for _ in range(self.num_enc_layers - 1)
                    ] if self.distil else None,
                    norm_layer=torch.nn.LayerNorm(f_d_model)
                ) for el in e_layers]

            encoder = EncoderStack(encoders, inp_lens)

        return encoder

    def get_custom_decoder(self, nhead, inner_classifier, patient):
        decoder_layer = DecoderLayer(self.d_model, nhead, self.d_ff)
        decoder_norm = LayerNorm(self.d_model)
        return Decoder(decoder_layer, self.num_dec_layers, norm=decoder_norm,
                       patient=patient, inner_classifier=inner_classifier)

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
        l_hand_memory = self.l_hand_encoder(src[0], mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        r_hand_memory = self.r_hand_encoder(src[1], mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        body_memory = self.body_encoder(src[2], mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        full_memory = torch.cat((l_hand_memory, r_hand_memory, body_memory), -1)

        output = self.decoder(tgt, full_memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask, training=training)
        return output


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x: [L, B, D/F] -> [B, D/F, L]
        x = x.permute(1, 2, 0)
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        # x: [B, D/F, L] -> [L//2, B, D/F]
        x = x.permute(2, 0, 1)
        # x = x.transpose(1,2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x: [L, B, D/F]

        new_x = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y)


class Encoder(nn.Module):
    def __init__(self, enc_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.enc_layers = nn.ModuleList(enc_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, mask=None, src_key_padding_mask=None):
        # x: [L, B, D/F]
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.enc_layers, self.conv_layers):
                x = attn_layer(x, attn_mask=mask)
                x = conv_layer(x)
            x = self.enc_layers[-1](x, attn_mask=mask)
        else:
            for attn_layer in self.enc_layers:
                x = attn_layer(x, attn_mask=mask)

        if self.norm is not None:
            x = self.norm(x)
        # x: [L//2, B, D/F]
        return x


class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, mask=None, src_key_padding_mask=None):
        # x: [L, B, F/D] -> [L', B, F/D]
        x_stack = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):  # inp_lens: 0, 1, 2
            inp_len = x.shape[0] // (2 ** i_len)
            x_s = encoder(x[-inp_len:, :, :])
            x_stack.append(x_s)
        x_stack = torch.cat(x_stack, 0)
        return x_stack


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
        # tgt: [1, B, F/D]
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

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None, tgt_is_causal: Optional[bool] = False,
                memory_is_causal: Optional[bool] = False) -> torch.Tensor:
        global is_dec_layer_checked
        if not is_dec_layer_checked:
            print('Using custom DecoderLayer')
            is_dec_layer_checked = True

        tgt = self.norm1(tgt + self.dropout1(tgt))
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt + self._ff_block(tgt))
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
            [42, 42, 24], [3, 3, 2, 9], selected_attn=attn_type, num_enc_layers=num_encoder_layers,
            num_dec_layers=num_decoder_layers,
            inner_classifiers=nn.ModuleList([
                nn.Linear(num_hid, num_classes) for _ in range(num_decoder_layers)
            ]),
            patient=1, use_pyramid_encoder=False, distil=False
        )
        self.class_query = nn.Parameter(torch.rand(1, 1, num_hid))
        self.projection = nn.Linear(num_hid, num_classes)

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
        out = self.projection(transformer_output).squeeze()
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
