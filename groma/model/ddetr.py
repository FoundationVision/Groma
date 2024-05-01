import json
import copy
import math
import torch
from torch import nn
from typing import List, Optional, Tuple, Union, Dict, Any

from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
    Dinov2Model,
    Dinov2Config,
    DeformableDetrConfig
)
from transformers.utils import logging
from transformers.models.deformable_detr.modeling_deformable_detr import DeformableDetrObjectDetectionOutput

from groma.model.ddetr_transformer import DeformableDetrTransformer

logger = logging.get_logger(__name__)


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class CustomDDETRConfig(PretrainedConfig):
    model_type = "ddetr"

    def __init__(
        self,
        vis_encoder_cfg=None,
        zs_weight_path=None,
        vis_output_layer=-1,
        ddetr_cfg=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vis_encoder_cfg is None:
            self.vis_encoder_cfg = Dinov2Config()
            logger.info("vis_encoder_config is None. initializing the Dinov2Config with default values.")
        elif isinstance(vis_encoder_cfg, dict):
            self.vis_encoder_cfg = Dinov2Config(**vis_encoder_cfg)
        elif isinstance(vis_encoder_cfg, Dinov2Config):
            self.vis_encoder_cfg = vis_encoder_cfg
        else:
            raise NotImplementedError("currently only supports Dinov2Model as vis_encoder.")

        if ddetr_cfg is None:
            self.ddetr_cfg = DeformableDetrConfig()
            logger.info("ddetr_cfg is None. Initializing the DeformableDetr with default values.")
        elif isinstance(ddetr_cfg, dict):
            self.ddetr_cfg = DeformableDetrConfig(**ddetr_cfg)
        elif isinstance(ddetr_cfg, DeformableDetrConfig):
            self.ddetr_cfg = ddetr_cfg
        else:
            raise NotImplementedError("currently only supports DeformableDetrTransformer as detector head.")

        self.zs_weight_path = zs_weight_path
        self.vis_output_layer = vis_output_layer

    def to_json_string(self, use_diff: bool = True) -> str:
        if use_diff is True:
            config_dict = copy.deepcopy(self)
            config_dict.vis_encoder_cfg = config_dict.vis_encoder_cfg.to_diff_dict()
            config_dict.ddetr_cfg = config_dict.ddetr_cfg.to_diff_dict()
            config_dict = config_dict.to_diff_dict()
        else:
            config_dict = copy.deepcopy(self)
            config_dict.vis_encoder_cfg = config_dict.vis_encoder_cfg.to_dict()
            config_dict.ddetr_cfg = config_dict.ddetr_cfg.to_dict()
            config_dict = config_dict.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"


class CustomDDETRModel(PreTrainedModel):
    config_class = CustomDDETRConfig

    def __init__(self, config: CustomDDETRConfig, pretrained_vis_encoder=None):
        super().__init__(config)

        if pretrained_vis_encoder is not None:
            self.vis_encoder = Dinov2Model.from_pretrained(pretrained_vis_encoder)
        else:
            self.vis_encoder = Dinov2Model(config.vis_encoder_cfg)

        self.ddetr_transformer = DeformableDetrTransformer(config.ddetr_cfg, config.zs_weight_path)

        num_feature_levels = config.ddetr_cfg.num_feature_levels
        in_channels = config.vis_encoder_cfg.hidden_size
        input_proj_list = []
        if num_feature_levels > 1:
            for i in range(num_feature_levels):
                if i == 0:
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, config.ddetr_cfg.d_model, kernel_size=3, stride=2, padding=1),
                        LayerNorm(config.ddetr_cfg.d_model),
                    ))
                elif i == 1:
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, config.ddetr_cfg.d_model, kernel_size=1),
                        LayerNorm(config.ddetr_cfg.d_model),
                        # nn.Conv2d(ddetr_cfg.d_model, ddetr_cfg.d_model, kernel_size=3, padding=1),
                    ))
                elif i == 2:
                    input_proj_list.append(nn.Sequential(
                        nn.ConvTranspose2d(in_channels, config.ddetr_cfg.d_model // 2, kernel_size=2, stride=2),
                        nn.Conv2d(config.ddetr_cfg.d_model // 2, config.ddetr_cfg.d_model, kernel_size=1),
                        LayerNorm(config.ddetr_cfg.d_model),
                        nn.Conv2d(config.ddetr_cfg.d_model, config.ddetr_cfg.d_model, kernel_size=3, padding=1),
                    ))
                elif i == 3:
                    input_proj_list.append(nn.Sequential(
                        nn.ConvTranspose2d(in_channels, config.ddetr_cfg.d_model // 2, kernel_size=2, stride=2),
                        LayerNorm(config.ddetr_cfg.d_model // 2),
                        nn.GELU(),
                        nn.ConvTranspose2d(config.ddetr_cfg.d_model // 2, config.ddetr_cfg.d_model // 4, kernel_size=2, stride=2),
                        nn.Conv2d(config.ddetr_cfg.d_model // 4, config.ddetr_cfg.d_model, kernel_size=1),
                        LayerNorm(config.ddetr_cfg.d_model),
                        nn.Conv2d(config.ddetr_cfg.d_model, config.ddetr_cfg.d_model, kernel_size=3, padding=1),
                    ))
                else:
                    raise ValueError("only support up to 4 feature levels!")
        else:
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, config.ddetr_cfg.d_model, kernel_size=1),
                LayerNorm(config.ddetr_cfg.d_model),
                # nn.Conv2d(ddetr_cfg.d_model, ddetr_cfg.d_model, kernel_size=3, padding=1),
            ))
        self.input_proj = nn.ModuleList(input_proj_list[::-1])
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def freeze_vis_encoder(self):
        self.vis_encoder.requires_grad_(False)

    def freeze_ddetr(self):
        self.ddetr_transformer.requires_grad_(False)

    def get_vis_encoder(self):
        return getattr(self, 'vis_encoder', None)

    def get_ddetr(self):
        return getattr(self, 'ddetr_transformer', None)

    def forward(
        self,
        images: Optional[list] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DeformableDetrObjectDetectionOutput]:
        with torch.no_grad():
            image_forward_outs = self.vis_encoder(images, output_hidden_states=True)
            select_hidden_state = torch.stack(image_forward_outs.hidden_states[-4:])
            image_features = torch.mean(select_hidden_state, dim=0)[:, 1:]
            bs, l, d = image_features.shape
            h = w = int(math.sqrt(l))
            assert h * w == l
            image_features = image_features.reshape(bs, h, w, d).permute(0, 3, 1, 2).contiguous()

        srcs = [input_proj(image_features) for input_proj in self.input_proj]
        masks = [torch.ones((src.shape[0],) + src.shape[2:], dtype=torch.bool, device=src.device) for src in srcs]

        return self.ddetr_transformer(
            sources=srcs,
            masks=masks,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


AutoConfig.register("ddetr", CustomDDETRConfig)
AutoModel.register(CustomDDETRConfig, CustomDDETRModel)
