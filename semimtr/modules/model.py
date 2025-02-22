import collections
import logging
import torch
import torch.nn as nn

from semimtr.utils.utils import CharsetMapper

_default_tfmer_cfg = dict(d_model=512, nhead=8, d_inner=2048,  # 1024
                          dropout=0.1, activation='relu')


class Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.max_length = config.dataset_max_length + 1
        self.charset = CharsetMapper(config.dataset_charset_path, max_length=self.max_length)

    def load(self, source, device=None, strict=True, submodule=None, exclude=None):
        state = torch.load(source, map_location=device)
        if source.endswith('.ckpt'):
            model_dict = state['state_dict']
            if list(model_dict.keys())[0].startswith('model.'):
                model_dict = collections.OrderedDict(
                    {k[6:]: v for k, v in model_dict.items() if k.startswith('model.')})
        else:
            model_dict = state
            if 'model' in model_dict:
                model_dict = model_dict['model']

        if submodule is None:
            out_features, old_num_classes = model_dict['proj.weight'].shape
            if old_num_classes != self.charset.num_classes:
                proj = nn.Linear(in_features=self.charset.num_classes, out_features=out_features, bias=False)
                cls = nn.Linear(in_features=out_features, out_features=self.charset.num_classes, bias=True)
                model_dict['proj.weight'] = proj.weight
                model_dict['cls.weight'] = cls.weight
                model_dict['cls.bias'] = cls.bias
            self.load_state_dict(model_dict, strict=strict)
        else:
            submodule_dict = collections.OrderedDict(
                {k.split('.', 1)[1]: v for k, v in model_dict.items()
                 if k.split('.', 1)[0] == submodule and k.split('.')[1] != exclude}
            )
            old_out_features, in_features = submodule_dict['cls.weight'].shape
            if old_out_features != self.charset.num_classes:
                old_out_features, in_features = submodule_dict['cls.weight'].shape
                cls = nn.Linear(in_features=in_features, out_features=self.charset.num_classes, bias=True)
                submodule_dict['cls.weight'] = cls.weight
                submodule_dict['cls.bias'] = cls.bias
            stat = self.load_state_dict(submodule_dict, strict=strict and exclude is None)  
            if stat.missing_keys or stat.unexpected_keys:
                logging.warning(f'Loading model with missing keys: {stat.missing_keys} '
                                f'and unexpected keys: {stat.unexpected_keys}')

    def _get_length(self, logit, dim=-1):
        """ Greed decoder to obtain length from logit"""
        out = (logit.argmax(dim=-1) == self.charset.null_label)
        abn = out.any(dim)
        out = ((out.cumsum(dim) == 1) & out).max(dim)[1]
        out = out + 1  # additional end token
        out = torch.where(abn, out, out.new_tensor(logit.shape[1]))
        return out

    @staticmethod
    def _get_padding_mask(length, max_length):
        length = length.unsqueeze(-1)
        grid = torch.arange(0, max_length, device=length.device).unsqueeze(0)
        return grid >= length

    @staticmethod
    def _get_square_subsequent_mask(sz, device, diagonal=0, fw=True):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device), diagonal=diagonal) == 1)
        if fw: mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @staticmethod
    def _get_location_mask(sz, device=None):
        mask = torch.eye(sz, device=device)
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask
