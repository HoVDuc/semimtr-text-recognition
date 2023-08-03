import torch
import torch.nn as nn

from semimtr.modules.model_vision import BaseVision
from semimtr.modules.model_language import BCNLanguage
from semimtr.modules.model_alignment import BaseAlignment
from semimtr.utils.utils import if_none, CharsetMapper


class ABINetIterModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.iter_size = if_none(config.model_iter_size, 1)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.charset = CharsetMapper(config.dataset_charset_path, max_length=self.max_length)
        self.vision = BaseVision(config)
        self.language = BCNLanguage(config)
        self.alignment = BaseAlignment(config)
        self.freeze_model()
        
    def freeze_model(self):
        for name, param in self.vision.named_parameters():
            if name == self.vision.backbone:
                param.requires_grad = False
        self.vision.cls = nn.Linear(in_features=512, out_features=self.charset.num_classes, bias=True)
    
        self.language.proj = nn.Linear(in_features=self.charset.num_classes, out_features=512, bias=True)
        self.language.cls = nn.Linear(in_features=512, out_features=self.charset.num_classes, bias=True)

    def forward(self, images, *args, **kwargs):
        v_res = self.vision(images, *args, **kwargs)
        a_res = v_res
        all_l_res, all_a_res = [], []
        for _ in range(self.iter_size):
            tokens = torch.softmax(a_res['logits'], dim=-1)
            lengths = a_res['pt_lengths']
            lengths.clamp_(2, self.max_length)  # TODO:move to langauge model
            samples = {'label': tokens, 'length': lengths}
            l_res = self.language(samples, *args, **kwargs)
            all_l_res.append(l_res)
            a_res = self.alignment(l_res['feature'], v_res['feature'], *args, **kwargs)
            all_a_res.append(a_res)
        if self.training:
            return all_a_res, all_l_res, v_res
        else:
            return a_res, all_l_res[-1], v_res
