from functools import reduce
import math
import operator

import numpy as np
from sklearn.preprocessing import maxabs_scale
import torch
from torch import nn


class AOTAugmentationPipeline:
    def __init__(self, is_conv=False):
        self.is_conv = is_conv

    def __call__(self, feat):
        # apply the transformation
        feat = np.array(feat, dtype=np.float32)
        # scale to -1 1
        # feat = maxabs_scale(feat, axis=0)
        aug_cond = np.zeros((9,), dtype=np.float32)

        if self.is_conv:
            feat = np.expand_dims(feat, axis=0)

        return feat, aug_cond


class AOTAugmentWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, input, sigma, aug_cond=None, mapping_cond=None, **kwargs):
        if aug_cond is None:
            aug_cond = input.new_zeros([input.shape[0], 9])
        if mapping_cond is None:
            mapping_cond = aug_cond
        else:
            mapping_cond = torch.cat([aug_cond, mapping_cond], dim=1)
        return self.inner_model(input, sigma, mapping_cond=mapping_cond, **kwargs)

    def set_skip_stages(self, skip_stages):
        return self.inner_model.set_skip_stages(skip_stages)

    def set_patch_size(self, patch_size):
        return self.inner_model.set_patch_size(patch_size)
