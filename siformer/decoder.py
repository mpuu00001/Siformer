import torch
import torch.nn.functional as F
from torch import Tensor, nn

from typing import Optional, Union, Callable

isChecked = False


class Decoder(nn.TransformerDecoder):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None, patient=1, inner_classifiers_config=None):
        super(Decoder, self).__init__(decoder_layer, num_layers, norm)
        self.patient = patient
        self.inner_classifiers = nn.ModuleList(
            [nn.Linear(inner_classifiers_config[0], inner_classifiers_config[1])
             for _ in range(num_layers)])

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
                _ = self.inner_classifiers[i](mod_output).squeeze()
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
                classifier_out = self.inner_classifiers[i](mod_output).squeeze()
                # labels = classifier_out.detach().argmax(dim=1)
                _, labels = torch.max(F.softmax(classifier_out, dim=1), 1)

                if patient_result is not None:
                    patient_out = patient_result.detach().argmax(dim=1)
                    _, patient_labels = torch.max(F.softmax(patient_out, dim=1), 1)

                    print(f"patient_labels: {patient_labels}")
                if (patient_result is not None) and torch.all(labels.eq(patient_labels)):
                    patient_counter += 1
                else:
                    patient_counter = 0

                patient_result = classifier_out
                if patient_counter == self.patient:
                    print("break")
                    break
            print(f"calculated_dec_layer_num: {calculated_layer_num}")

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
