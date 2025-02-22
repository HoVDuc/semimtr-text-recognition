import logging
import torch.nn as nn

from semimtr.modules.model import _default_tfmer_cfg
from semimtr.modules.model import Model
from semimtr.modules.transformer import (PositionalEncoding,
                                         TransformerDecoder,
                                         TransformerDecoderLayer)
from semimtr.utils.utils import if_none


class BCNLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = if_none(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = if_none(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = if_none(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = if_none(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = if_none(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = if_none(config.model_language_num_layers, 4)
        num_classes = self.charset.num_classes
        self.d_model = d_model
        self.detach = if_none(config.model_language_detach, True)
        self.use_self_attn = if_none(config.model_language_use_self_attn, False)
        self.loss_weight = if_none(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = if_none(config.global_debug, False)

        self.proj = nn.Linear(num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_inner, dropout,
                                                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint)

    def forward(self, samples, *args, **kwargs):
        """
        Args:
            samples: dict
                tokens: (N, T, C) where T is length, N is batch size and C is classes number
                lengths: (N,)
        """
        tokens, lengths = samples['label'], samples['length']
        if self.detach: tokens = tokens.detach()
        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)
        output = self.model(qeury, embed,
                            tgt_key_padding_mask=padding_mask,
                            memory_mask=location_mask,
                            memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res = {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
               'loss_weight': self.loss_weight, 'name': 'language'}
        return res
