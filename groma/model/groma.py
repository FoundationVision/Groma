import json
import copy
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union, Dict, Any

from mmcv.ops.nms import nms
from torchvision.ops import box_iou
from transformers.utils import logging
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.image_transforms import center_to_corners_format
from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM
)

from groma.model.ddetr import CustomDDETRConfig, CustomDDETRModel
from groma.model.roi_align import MLVLROIQueryModule
from groma.constants import DEFAULT_TOKENS, REGION_IDX_TOKENS, IGNORE_INDEX

logger = logging.get_logger(__name__)


class GromaConfig(PretrainedConfig):
    model_type = "groma"

    def __init__(
        self,
        llm_cfg=None,
        perceiver_cfg=None,
        num_new_token=0,
        nms_thres=0.0,
        box_score_thres=0.0,
        max_region_num=100,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if perceiver_cfg is None:
            self.perceiver_cfg = CustomDDETRConfig()
            logger.info("perceiver_cfg is None. initializing the CustomDDETRConfig with default values.")
        elif isinstance(perceiver_cfg, dict):
            self.perceiver_cfg = CustomDDETRConfig(**perceiver_cfg)
        elif isinstance(perceiver_cfg, CustomDDETRConfig):
            self.perceiver_cfg = perceiver_cfg
        else:
            raise NotImplementedError("currently only supports CustomDDETR as perceiver.")

        if llm_cfg is None:
            self.llm_cfg = LlamaConfig()
            logger.info("llm_cfg is None. Initializing the LlamaModel with default values.")
        elif isinstance(llm_cfg, dict):
            self.llm_cfg = LlamaConfig(**llm_cfg)
        elif isinstance(llm_cfg, LlamaConfig):
            self.llm_cfg = llm_cfg
        else:
            raise NotImplementedError("currently only supports LlamaModel as LLM.")

        self.nms_thres = nms_thres
        self.box_score_thres = box_score_thres
        self.max_region_num = max_region_num
        self.num_new_token = num_new_token
        self.vocab_size = self.llm_cfg.vocab_size + num_new_token

    def to_json_string(self, use_diff: bool = True) -> str:
        if use_diff:
            config_dict = copy.deepcopy(self)
            config_dict.perceiver_cfg = config_dict.perceiver_cfg.to_diff_dict()
            config_dict.llm_cfg = config_dict.llm_cfg.to_diff_dict()
            config_dict = config_dict.to_diff_dict()
        else:
            config_dict = copy.deepcopy(self)
            config_dict.perceiver_cfg = config_dict.perceiver_cfg.to_dict()
            config_dict.llm_cfg = config_dict.llm_cfg.to_dict()
            config_dict = config_dict.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"


class GromaModel(PreTrainedModel):
    config_class = GromaConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: GromaConfig,
        pretrained_perceiver: Optional[str] = None,
        pretrained_llm: Optional[str] = None
    ):
        super().__init__(config)
        if pretrained_perceiver is None:
            self.perceiver = CustomDDETRModel(config.perceiver_cfg)
        else:
            self.perceiver = CustomDDETRModel.from_pretrained(pretrained_perceiver)
            config.perceiver_cfg = CustomDDETRConfig.from_pretrained(pretrained_perceiver)

        if pretrained_llm is None:
            self.llm = LlamaForCausalLM(config.llm_cfg)
        else:
            self.llm = LlamaForCausalLM.from_pretrained(pretrained_llm)
            config.llm_cfg = LlamaConfig.from_pretrained(pretrained_llm)
            config.vocab_size = config.llm_cfg.vocab_size + config.num_new_token

        image_embed_dim = self.perceiver.config.vis_encoder_cfg.hidden_size
        text_embed_dim = self.llm.config.hidden_size
        self.img_txt_bridge = nn.Sequential(
            nn.Linear(image_embed_dim * 4, text_embed_dim),
            nn.GELU(),
            nn.Linear(text_embed_dim, text_embed_dim),
        )
        self.region_encoder = MLVLROIQueryModule(image_embed_dim, text_embed_dim, num_levels=3)

        self.extra_lm_head = nn.Linear(text_embed_dim, config.num_new_token, bias=False)

        self.new_input_embs = nn.Embedding(config.num_new_token, text_embed_dim)
        input_embeds = self.llm.get_input_embeddings().weight.data
        input_embeds_avg = input_embeds[:].mean(dim=0, keepdim=True)
        self.new_input_embs.weight.data[:, :] = input_embeds_avg

        self.pad_token_id = None
        self.img_token_id = None
        self.reg_token_id = None
        self.refer_box_token_id = None
        self.refer_feat_token_id = None
        self.ground_box_token_id = None
        self.box_idx_token_ids = None

        self.post_init()

    def init_special_token_id(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        self.img_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['image']])[0]
        self.reg_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['region']])[0]
        self.refer_box_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['rbox']])[0]
        self.refer_feat_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['rfeat']])[0]
        self.ground_box_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['gbox']])[0]
        self.box_idx_token_ids = tokenizer.convert_tokens_to_ids(REGION_IDX_TOKENS)
        return

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value

    def freeze_perceiver(self):
        self.perceiver.requires_grad_(False)

    def freeze_llm(self):
        self.llm.requires_grad_(False)

    def freeze_vl_bridge(self):
        self.vl_bridge.requires_grad_(False)

    def get_perceiver(self):
        return getattr(self, 'perceiver', None)

    def get_llm(self):
        return getattr(self, 'llm', None)

    def get_input_embeddings(self, input_ids):
        ori_embed_tokens = self.llm.get_input_embeddings()
        mask = input_ids >= ori_embed_tokens.num_embeddings
        ori_ids = input_ids.masked_fill(mask, 0)
        new_ids = input_ids - ori_embed_tokens.num_embeddings
        new_ids = new_ids.masked_fill(~mask, 0)
        input_embeddings = ori_embed_tokens(ori_ids)
        new_input_embeddings = self.new_input_embs(new_ids)
        input_embeddings[mask] = new_input_embeddings[mask]
        return input_embeddings

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache"),
            "images": kwargs.get("images", None),
            "refer_boxes": kwargs.get("refer_boxes", None),
            "ground_boxes": kwargs.get("ground_boxes", None),
        })
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[list] = None,
        refer_boxes: Optional[list] = None,
        ground_boxes: Optional[list] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        vis_outputs = None
        if past_key_values is None:
            with torch.no_grad():
                # encode image
                vis_encoder_outs = self.perceiver.vis_encoder(images, output_hidden_states=True)
                select_hidden_state_layer = getattr(self.perceiver.config, "vis_output_layer", -1)
                image_features = vis_encoder_outs.hidden_states[select_hidden_state_layer][:, 1:]

                # downsample feature map
                bs, l, d = image_features.shape
                h = w = int(math.sqrt(l))
                assert h * w == l, "input image shape must be square"
                image_features = image_features.reshape(bs, h, w, d)
                image_features = torch.cat([
                    image_features[:, 0::2, 0::2, :],
                    image_features[:, 1::2, 0::2, :],
                    image_features[:, 0::2, 1::2, :],
                    image_features[:, 1::2, 1::2, :],
                ], dim=-1)
                image_features = image_features.reshape(bs, l // 4, d * 4)

                # predict boxes and objectness scores
                ddetr_inputs = torch.stack(vis_encoder_outs.hidden_states[-4:])
                ddetr_inputs = torch.mean(ddetr_inputs, dim=0)[:, 1:]
                ddetr_inputs = ddetr_inputs.reshape(bs, h, w, d).permute(0, 3, 1, 2).contiguous()
                srcs = [input_proj(ddetr_inputs) for input_proj in self.perceiver.input_proj]
                masks = [torch.ones((src.shape[0],) + src.shape[2:], dtype=torch.bool, device=src.device) for src in srcs]
                ddetr_outs = self.perceiver.ddetr_transformer(srcs, masks, return_dict=True)
                pred_boxes = ddetr_outs.pred_boxes
                scores_coco = ddetr_outs.logits['coco'].squeeze(dim=-1).sigmoid()
                scores_sa1b = ddetr_outs.logits['sa1b'].squeeze(dim=-1).sigmoid()
                scores_fused = scores_coco ** 0.4 * scores_sa1b ** 0.6

                # select regions to encode
                selected_boxes = []
                if refer_boxes is None:
                    refer_boxes = [torch.empty((0, 4), device=pred_boxes.device) for _ in range(bs)]
                if ground_boxes is None:
                    ground_boxes = [torch.empty((0, 4), device=pred_boxes.device) for _ in range(bs)]
                for i in range(bs):
                    # merge predicted boxes and user provided (referring) boxes
                    scores_refer = torch.ones(refer_boxes[i].shape[0]).to(scores_fused.device)
                    # inject ground-truth boxes to predicted boxes, but assign low scores to gt_boxes
                    # if there exists close predicted boxes, gt boxes will be filter out by NMS
                    scores_ground = torch.ones(ground_boxes[i].shape[0]).to(scores_fused.device) * 0.2
                    scores = torch.cat((scores_fused[i], scores_refer, scores_ground))
                    input_boxes = torch.cat((pred_boxes[i], refer_boxes[i], ground_boxes[i]))
                    # filter out redundant and low-quality boxes, select top-k boxes
                    nms_inds = nms(
                        scores=scores,
                        boxes=center_to_corners_format(input_boxes),
                        iou_threshold=self.config.nms_thres,
                        score_threshold=self.config.box_score_thres,
                        max_num=self.config.max_region_num
                    )[-1]
                    if len(nms_inds) > 0:
                        input_boxes = input_boxes[nms_inds]
                        rand_inds = torch.randperm(len(input_boxes))
                        input_boxes = input_boxes[rand_inds]
                    else:
                        max_ind = torch.max(scores, dim=0).indices
                        input_boxes = input_boxes[max_ind: max_ind + 1]
                    selected_boxes.append(input_boxes)

            # replace placeholder of refer/ground box with true index
            refer_box_inds = []
            for i in range(bs):
                if self.refer_box_token_id in input_ids[i]:
                    ious = box_iou(
                        center_to_corners_format(refer_boxes[i]),
                        center_to_corners_format(selected_boxes[i])
                    )
                    matched_inds = torch.max(ious, dim=-1).indices
                    refer_box_inds.append(matched_inds)
                    box_idx_token_ids = torch.tensor(self.box_idx_token_ids).to(matched_inds.device)
                    matched_box_idxs = box_idx_token_ids[matched_inds]
                    mask = input_ids[i] == self.refer_box_token_id
                    input_ids[i].masked_scatter_(mask, matched_box_idxs)
                else:
                    refer_box_inds.append([])
                if self.ground_box_token_id in input_ids[i]:
                    ious = box_iou(
                        center_to_corners_format(ground_boxes[i]),
                        center_to_corners_format(selected_boxes[i])
                    )
                    matched_inds = torch.max(ious, dim=-1).indices
                    box_idx_token_ids = torch.tensor(self.box_idx_token_ids).to(matched_inds.device)
                    matched_box_idxs = box_idx_token_ids[matched_inds]
                    mask = input_ids[i] == self.ground_box_token_id
                    input_ids[i].masked_scatter_(mask, matched_box_idxs)
                    labels[i].masked_scatter_(mask, matched_box_idxs)
            assert len(refer_box_inds) == bs

            # encode region features
            mlvl_feats = vis_encoder_outs.hidden_states[-3:]
            mlvl_feats = [mlvl_feat[:, 1:] for mlvl_feat in mlvl_feats]
            region_features = self.region_encoder(mlvl_feats, selected_boxes)
            refer_region_features = [region_feat[ind] for region_feat, ind in zip(region_features, refer_box_inds)]

            # inject placeholder for visual input to input_ids and labels
            num_image_tokens = image_features.shape[1]
            num_region_tokens = [x.shape[0] for x in region_features]
            new_input_ids = []
            new_labels = []
            for i in range(bs):
                assert self.img_token_id in input_ids[i] and self.reg_token_id in input_ids[i]
                img_token_pos = (input_ids[i] == self.img_token_id).nonzero(as_tuple=True)[0]
                reg_token_pos = (input_ids[i] == self.reg_token_id).nonzero(as_tuple=True)[0]
                pad_token_pos = (input_ids[i] == self.pad_token_id).nonzero(as_tuple=True)[0]
                pad_token_pos = pad_token_pos[0] if len(pad_token_pos) > 0 else len(input_ids[i])
                assert img_token_pos < reg_token_pos
                img_placeholder = torch.full((num_image_tokens,), self.img_token_id).to(input_ids.device)
                reg_placeholder = [
                    torch.tensor([self.box_idx_token_ids[j], self.reg_token_id]) for j in range(num_region_tokens[i])]
                reg_placeholder = torch.cat(reg_placeholder).to(input_ids.device)
                new_input_ids.append(torch.cat((
                    input_ids[i][:img_token_pos],
                    img_placeholder,
                    input_ids[i][img_token_pos + 1: reg_token_pos],
                    reg_placeholder,
                    input_ids[i][reg_token_pos + 1: pad_token_pos]
                )))
                if labels is not None:
                    new_labels.append(torch.cat((
                        labels[i][:img_token_pos],
                        torch.full((num_image_tokens,), IGNORE_INDEX, device=labels.device),
                        labels[i][img_token_pos + 1: reg_token_pos],
                        torch.full((num_region_tokens[i] * 2,), IGNORE_INDEX, device=labels.device),
                        labels[i][reg_token_pos + 1: pad_token_pos]
                    )))
            input_ids = torch.nn.utils.rnn.pad_sequence(
                new_input_ids,
                batch_first=True,
                padding_value=self.pad_token_id)
            if labels is not None:
                labels = torch.nn.utils.rnn.pad_sequence(
                    new_labels,
                    batch_first=True,
                    padding_value=IGNORE_INDEX)
            attention_mask = input_ids.ne(self.pad_token_id)

            # inject visual input
            inputs_embeds = self.get_input_embeddings(input_ids)
            image_features = self.img_txt_bridge(image_features).to(inputs_embeds.dtype)
            region_features = torch.cat(region_features).to(inputs_embeds.dtype)
            refer_region_features = torch.cat(refer_region_features).to(inputs_embeds.dtype)
            img_mask = input_ids == self.img_token_id
            inputs_embeds.masked_scatter_(img_mask[:, :, None], image_features)
            reg_mask = input_ids == self.reg_token_id
            inputs_embeds.masked_scatter_(reg_mask[:, :, None], region_features)
            ref_mask = input_ids == self.refer_feat_token_id
            inputs_embeds.masked_scatter_(ref_mask[:, :, None], refer_region_features)

            vis_outputs = {
                'pred_boxes': selected_boxes,
                'image_features': image_features,
                'region_features': region_features.detach()
            }
        else:
            bs = past_key_values[0][0].shape[0]
            token_length = past_key_values[0][0].shape[-2] + 1
            attention_mask = torch.ones((bs, token_length), device=attention_mask.device)

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)

        output_attentions = output_attentions if output_attentions is not None else self.config.llm_cfg.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.llm_cfg.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.llm_cfg.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.llm.model(
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.llm.lm_head(hidden_states)
        extra_logits = self.extra_lm_head(hidden_states)
        logits = torch.concat((logits, extra_logits), dim=-1)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]  # + (pred_boxes,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=(outputs.hidden_states, vis_outputs),
            attentions=outputs.attentions,
        )


AutoConfig.register("groma", GromaConfig)
AutoModel.register(GromaConfig, GromaModel)
