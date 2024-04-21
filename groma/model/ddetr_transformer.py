import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, normal_
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrPreTrainedModel,
    DeformableDetrConfig,
    DeformableDetrEncoder,
    DeformableDetrDecoder,
    DeformableDetrDecoderLayer,
    DeformableDetrMLPPredictionHead,
    DeformableDetrModelOutput,
    DeformableDetrHungarianMatcher,
    DeformableDetrLoss,
    DeformableDetrObjectDetectionOutput,
    DeformableDetrDecoderOutput,
    DeformableDetrMultiscaleDeformableAttention,
    build_position_encoding,
    generalized_box_iou,
    _get_clones,
    inverse_sigmoid
)
from transformers.pytorch_utils import meshgrid
from transformers.file_utils import requires_backends
from transformers.image_transforms import center_to_corners_format
from scipy.optimize import linear_sum_assignment
from mmcv.ops.bbox import bbox_overlaps


class ZeroShotClassifier(nn.Module):
    def __init__(
        self,
        zs_weight_path: str,
        input_size: int = 512,
        norm_weight: bool = True,
        bias: float = 0.0,
        norm_temperature: float = 50.0
    ):
        super().__init__()
        zs_weight = torch.load(zs_weight_path).T
        zs_weight_dim, num_classes = zs_weight.shape
        zs_weight = F.normalize(zs_weight, p=2, dim=0) if norm_weight else zs_weight

        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature
        self.linear = nn.Linear(input_size, zs_weight_dim)
        self.zs_weight = nn.Embedding.from_pretrained(zs_weight, freeze=True)
        self.bias = nn.Parameter(torch.ones(num_classes) * bias)

    def forward(self, x):
        x = self.linear(x)
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=1)
        x = x @ self.zs_weight.weight + self.bias
        return x


class DeformableDetrDecoderX(DeformableDetrDecoder):
    def __init__(self, config: DeformableDetrConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layers = nn.ModuleList([DeformableDetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.gradient_checkpointing = False

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed_enc = None
        self.class_embed_coco = None
        self.class_embed_sa1b = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        intermediate = ()
        intermediate_reference_points = ()

        for idx, decoder_layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                        reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                if reference_points.shape[-1] != 2:
                    raise ValueError("Reference points' last dimension must be of size 2")
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    encoder_hidden_states=encoder_hidden_states,
                    reference_points=reference_points_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[idx](hidden_states)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    if reference_points.shape[-1] != 2:
                        raise ValueError(
                            f"Reference points' last dimension must be of size 2, but is {reference_points.shape[-1]}"
                        )
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                # reference_points = new_reference_points.detach()

            intermediate += (hidden_states,)
            intermediate_reference_points += (new_reference_points,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_reference_points,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return DeformableDetrDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class BoxOnlyHungarianMatcher(nn.Module):
    def __init__(self, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()
        requires_backends(self, ["scipy"])

        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, pred_boxes, gt_boxes):
        batch_size, num_queries = pred_boxes.shape[:2]

        # We flatten to compute the cost matrices in a batch
        pred_boxes = pred_boxes.flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        target_boxes = torch.cat(gt_boxes)

        # Compute the L1 cost between boxes
        bbox_cost = torch.cdist(pred_boxes, target_boxes, p=1)

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(pred_boxes), center_to_corners_format(target_boxes))

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v) for v in gt_boxes]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class DeformableDetrLossX(DeformableDetrLoss):
    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`List[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # (Niels): comment out function below, distributed training to be added
        # if is_dist_avail_and_initialized():
        # torch.distributed.all_reduce(num_boxes)
        # (Niels) in original implementation, num_boxes is divided by get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["class_labels"] = torch.zeros_like(bt["class_labels"])
            indices = self.matcher(enc_outputs, bin_targets)

            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class DeformableDetrTransformer(DeformableDetrPreTrainedModel):
    def __init__(self, config: DeformableDetrConfig, zs_weight_path=None):
        super().__init__(config)

        self.encoder = DeformableDetrEncoder(config)
        self.decoder = DeformableDetrDecoderX(config)

        self.position_encoding = build_position_encoding(config)
        self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, config.d_model))

        if not config.two_stage:
            self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model * 2)
        else:
            self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)

        if config.two_stage:
            self.enc_output = nn.Linear(config.d_model, config.d_model)
            self.enc_output_norm = nn.LayerNorm(config.d_model)
            self.pos_trans = nn.Linear(config.d_model * 2, config.d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(config.d_model * 2)
        else:
            self.reference_points = nn.Linear(config.d_model, 2)

        self._reset_parameters()

        # Detection heads on top
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        self.class_embed_enc = nn.Linear(config.d_model, config.num_labels)
        self.class_embed_coco = nn.Linear(config.d_model, config.num_labels)
        self.class_embed_sa1b = nn.Linear(config.d_model, config.num_labels)
        self.class_embed_enc.bias.data = torch.ones(config.num_labels, device=self.device) * bias_value
        self.class_embed_coco.bias.data = torch.ones(config.num_labels, device=self.device) * bias_value
        self.class_embed_sa1b.bias.data = torch.ones(config.num_labels, device=self.device) * bias_value

        self.bbox_embed = DeformableDetrMLPPredictionHead(
            input_dim=config.d_model, hidden_dim=256, output_dim=4, num_layers=3
        )
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        # num_pred = (config.decoder_layers + 1) if config.two_stage else config.decoder_layers
        num_pred = config.decoder_layers
        if config.with_box_refine:
            self.class_embed_coco = _get_clones(self.class_embed_coco, num_pred)
            self.class_embed_sa1b = _get_clones(self.class_embed_sa1b, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred + 1)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed_coco = nn.ModuleList([self.class_embed_coco for _ in range(num_pred)])
            self.class_embed_sa1b = nn.ModuleList([self.class_embed_sa1b for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.decoder.bbox_embed = None
        if config.two_stage:
            # hack implementation for two-stage
            self.decoder.class_embed_enc = self.class_embed_enc
            self.decoder.class_embed_coco = self.class_embed_coco
            self.decoder.class_embed_sa1b = self.class_embed_sa1b
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, DeformableDetrMultiscaleDeformableAttention):
                m._reset_parameters()
        if not self.config.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        """Get the valid ratio of all feature maps."""
        _, height, width = mask.shape
        valid_height = torch.sum(mask[:, :, 0], 1)
        valid_width = torch.sum(mask[:, 0, :], 1)
        valid_ratio_heigth = valid_height.float() / height
        valid_ratio_width = valid_width.float() / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    def gen_encoder_output_proposals(self, enc_output, padding_mask, spatial_shapes):
        """Generate the encoder output proposals from encoded enc_output.

        Args:
            enc_output (Tensor[batch_size, sequence_length, hidden_size]): Output of the encoder.
            padding_mask (Tensor[batch_size, sequence_length]): Padding mask for `enc_output`.
            spatial_shapes (Tensor[num_feature_levels, 2]): Spatial shapes of the feature maps.

        Returns:
            `tuple(torch.FloatTensor)`: A tuple of feature map and bbox prediction.
                - object_query (Tensor[batch_size, sequence_length, hidden_size]): Object query features. Later used to
                  directly predict a bounding box. (without the need of a decoder)
                - output_proposals (Tensor[batch_size, sequence_length, 4]): Normalized proposals, after an inverse
                  sigmoid.
        """
        batch_size = enc_output.shape[0]
        proposals = []
        _cur = 0
        for level, (height, width) in enumerate(spatial_shapes):
            mask_flatten_ = padding_mask[:, _cur: (_cur + height * width)].view(batch_size, height, width, 1)
            valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = meshgrid(
                torch.linspace(0, height - 1, height, dtype=torch.float32, device=enc_output.device),
                torch.linspace(0, width - 1, width, dtype=torch.float32, device=enc_output.device),
                indexing="ij",
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).view(batch_size, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale
            width_heigth = torch.ones_like(grid) * 0.05 * (2.0 ** level)
            proposal = torch.cat((grid, width_heigth), -1).view(batch_size, -1, 4)
            proposals.append(proposal)
            _cur += height * width
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))  # inverse sigmoid
        output_proposals = output_proposals.masked_fill(padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        # assign each pixel as an object query
        object_query = enc_output
        object_query = object_query.masked_fill(padding_mask.unsqueeze(-1), float(0))
        object_query = object_query.masked_fill(~output_proposals_valid, float(0))
        object_query = self.enc_output_norm(self.enc_output(object_query))
        return object_query, output_proposals

    def get_proposal_pos_embed(self, proposals, num_pos_feats=512):
        """Get the position embedding of the proposals."""

        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
        # batch_size, num_queries, 4
        proposals = proposals.sigmoid() * scale
        # batch_size, num_queries, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # batch_size, num_queries, 4, 64, 2 -> batch_size, num_queries, 512
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        outputs_class = outputs_class.transpose(0, 1)
        outputs_coord = outputs_coord.transpose(0, 1)
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def _get_source_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    def class_agnostic_nms(self, batched_boxes, thres=0.9):
        batched_boxes = center_to_corners_format(batched_boxes)
        batched_ious = [bbox_overlaps(boxes, boxes) for boxes in batched_boxes]
        nms_mask = torch.ones(batched_boxes.shape[:2], device=batched_boxes.device)
        for batch_ind, ious in enumerate(batched_ious):
            row, col = len(ious), len(ious[0])
            overlaped_pairs = [(i, j) for i in range(row) for j in range(i+1, col) if ious[i][j] > thres]
            masked_inds = []
            for i, j in overlaped_pairs:
                if i not in masked_inds and j not in masked_inds:
                    masked_inds.append(j)
            nms_mask[batch_ind, masked_inds] = 0.
        return nms_mask

    def box_area_filter(self, batched_boxes, thres=0.005):
        area_mask = torch.ones(batched_boxes.shape[:2], device=batched_boxes.device)
        box_areas = [[x[2] * x[3] for x in boxes] for boxes in batched_boxes]
        box_areas = torch.tensor(box_areas, device=batched_boxes.device)
        area_mask[box_areas < thres] = 0.
        return area_mask

    def extract_feature(
        self,
        sources,
        masks,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        position_embeddings_list = []
        for source, mask in zip(sources, masks):
            position_embeddings = self.position_encoding(source, mask).to(source.dtype)
            position_embeddings_list.append(position_embeddings)

        # Create queries
        query_embeds = self.query_position_embeddings.weight

        # Prepare encoder inputs (by flattening)
        source_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for level, (source, mask, pos_embed) in enumerate(zip(sources, masks, position_embeddings_list)):
            batch_size, num_channels, height, width = source.shape
            spatial_shape = (height, width)
            spatial_shapes.append(spatial_shape)
            source = source.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            source_flatten.append(source)
            mask_flatten.append(mask)
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        valid_ratios = valid_ratios.float()

        # Fourth, sent source_flatten + mask_flatten + lvl_pos_embed_flatten (backbone + proj layer output) through encoder
        # Also provide spatial_shapes, level_start_index and valid_ratios
        encoder_outputs = self.encoder(
            inputs_embeds=source_flatten,
            attention_mask=mask_flatten,
            position_embeddings=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Fifth, prepare decoder inputs
        batch_size, _, num_channels = encoder_outputs[0].shape
        enc_outputs_class = None
        enc_outputs_coord_logits = None
        if self.config.two_stage:
            object_query_embedding, output_proposals = self.gen_encoder_output_proposals(
                encoder_outputs[0], ~mask_flatten, spatial_shapes
            )
            enc_outputs_class = self.decoder.class_embed_enc(object_query_embedding)
            delta_bbox = self.decoder.bbox_embed[-1](object_query_embedding)
            enc_outputs_coord_logits = delta_bbox + output_proposals

            # only keep top scoring `config.two_stage_num_proposals` proposals
            topk = self.config.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_logits = torch.gather(
                enc_outputs_coord_logits, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )

            topk_coords_logits = topk_coords_logits.detach()
            reference_points = topk_coords_logits.sigmoid()
            init_reference_points = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(
                self.get_proposal_pos_embed(topk_coords_logits, num_pos_feats=num_channels // 2)))
            # query_embed, target = torch.split(pos_trans_out, num_channels, dim=2)
            target = query_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            query_embed, _ = torch.split(pos_trans_out, num_channels, dim=2)
        else:
            query_embed, target = torch.split(query_embeds, num_channels, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(batch_size, -1, -1)
            target = target.unsqueeze(0).expand(batch_size, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_points = reference_points

        decoder_outputs = self.decoder(
            inputs_embeds=target,
            position_embeddings=query_embed,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            enc_outputs = tuple(value for value in [enc_outputs_class, enc_outputs_coord_logits] if value is not None)
            tuple_outputs = (init_reference_points,) + decoder_outputs + encoder_outputs + enc_outputs

            return tuple_outputs

        return DeformableDetrModelOutput(
            init_reference_points=init_reference_points,
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
        )

    def forward_box(
        self,
        decoder_outputs,
        target_boxes=None,
    ):
        hidden_states = decoder_outputs.intermediate_hidden_states
        inter_references = decoder_outputs.intermediate_reference_points

        last_level = hidden_states.shape[1] - 1
        reference = inter_references[:, last_level-1]
        reference = inverse_sigmoid(reference)
        delta_bbox = self.bbox_embed[last_level](hidden_states[:, last_level])
        outputs_coord_logits = delta_bbox + reference
        pred_boxes = outputs_coord_logits.sigmoid()

        outputs_class = self.class_embed[last_level](hidden_states[:, last_level])
        score = torch.max(outputs_class.sigmoid(), dim=-1).values

        nms_mask = self.class_agnostic_nms(pred_boxes, thres=0.9)
        size_mask = self.box_area_filter(pred_boxes, thres=0.005)
        if all(torch.sum(nms_mask * size_mask, dim=-1) >= 12):
            score *= nms_mask * size_mask
        elif all(torch.sum(nms_mask, dim=-1) >= 12):
            score *= nms_mask
        selected_idx = torch.topk((score * nms_mask), k=12, dim=-1, sorted=False).indices
        pred_boxes = torch.stack([pred_boxes[i, idx] for i, idx in enumerate(selected_idx)])

        loss_loc, matched_idx = None, None
        if target_boxes is not None:
            matcher = BoxOnlyHungarianMatcher(bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost)
            indices = matcher(pred_boxes, target_boxes)

            num_boxes = sum(len(t) for t in target_boxes)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=pred_boxes.device)
            num_boxes = torch.clamp(num_boxes, min=1).item()

            idx = self._get_source_permutation_idx(indices)
            source_boxes = pred_boxes[idx]
            target_boxes = torch.cat([t[i] for t, (_, i) in zip(target_boxes, indices)], dim=0)

            loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")
            loss_bbox = loss_bbox.sum() / num_boxes
            loss_giou = 1 - torch.diag(generalized_box_iou(
                center_to_corners_format(source_boxes),
                center_to_corners_format(target_boxes))
            )
            loss_giou = loss_giou.sum() / num_boxes
            loss_loc = self.config.bbox_loss_coefficient * loss_bbox + self.config.giou_loss_coefficient * loss_giou

            src_indices = [x[0] for x in indices]
            tgt_indices = [x[1] for x in indices]
            permute_indices = [sorted(range(len(x)), key=lambda k: x[k]) for x in tgt_indices]
            src_permuted = [x[i] for i, x in zip(permute_indices, src_indices)]
            matched_idx = torch.cat(src_permuted)

        return pred_boxes, selected_idx, matched_idx, loss_loc

    def forward(
        self,
        sources,
        masks,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.extract_feature(
            sources,
            masks,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        hidden_states = outputs.intermediate_hidden_states if return_dict else outputs[2]
        init_reference = outputs.init_reference_points if return_dict else outputs[0]
        inter_references = outputs.intermediate_reference_points if return_dict else outputs[3]

        # class logits + predicted bounding boxes
        outputs_classes_coco = []
        outputs_classes_sa1b = []
        outputs_coords = []

        for level in range(hidden_states.shape[1]):
            if level == 0:
                reference = init_reference
            else:
                reference = inter_references[:, level - 1]
            reference = inverse_sigmoid(reference)
            outputs_class_coco = self.class_embed_coco[level](hidden_states[:, level])
            outputs_class_sa1b = self.class_embed_sa1b[level](hidden_states[:, level])
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])
            if reference.shape[-1] == 4:
                outputs_coord_logits = delta_bbox + reference
            elif reference.shape[-1] == 2:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
            else:
                raise ValueError(f"reference.shape[-1] should be 4 or 2, but got {reference.shape[-1]}")
            outputs_coord = outputs_coord_logits.sigmoid()
            outputs_classes_coco.append(outputs_class_coco)
            outputs_classes_sa1b.append(outputs_class_sa1b)
            outputs_coords.append(outputs_coord)
        # Keep batch_size as first dimension
        outputs_class_coco = torch.stack(outputs_classes_coco, dim=1)
        outputs_class_sa1b = torch.stack(outputs_classes_sa1b, dim=1)
        outputs_coord = torch.stack(outputs_coords, dim=1)
        outputs_class = {
            'coco': outputs_class_coco,
            'sa1b': outputs_class_sa1b
        }
        logits = {
            'coco': outputs_class_coco[:, -1],
            'sa1b': outputs_class_sa1b[:, -1]
        }
        pred_boxes = outputs_coord[:, -1]

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            matcher = DeformableDetrHungarianMatcher(
                class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost
            )
            losses = ["labels", "boxes", "cardinality"]
            criterion = DeformableDetrLossX(
                matcher=matcher,
                num_classes=self.config.num_labels,
                focal_alpha=self.config.focal_alpha,
                losses=losses,
            ).to(self.device)

            source = labels[0]['source']
            outputs_loss = dict()
            outputs_loss["logits"] = logits[source]
            outputs_loss["pred_boxes"] = pred_boxes
            if self.config.auxiliary_loss:
                auxiliary_outputs = self._set_aux_loss(outputs_class[source], outputs_coord)
                outputs_loss["auxiliary_outputs"] = auxiliary_outputs
            if self.config.two_stage:
                enc_outputs_coord = outputs.enc_outputs_coord_logits.sigmoid()
                outputs_loss["enc_outputs"] = {"logits": outputs.enc_outputs_class, "pred_boxes": enc_outputs_coord}

            loss_dict = criterion(outputs_loss, labels)

            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {
                "loss_ce": self.config.cls_loss_coefficient,
                "loss_bbox": self.config.bbox_loss_coefficient,
                "loss_giou": self.config.giou_loss_coefficient
            }
            if self.config.auxiliary_loss:
                aux_weight_dict = {}
                for i in range(self.config.decoder_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)
            if self.config.two_stage:
                weight_dict["loss_ce_enc"] = self.config.cls_loss_coefficient
                weight_dict["loss_bbox_enc"] = self.config.bbox_loss_coefficient
                weight_dict["loss_giou_enc"] = self.config.giou_loss_coefficient
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes) + outputs
            tuple_outputs = ((loss, loss_dict) + output) if loss is not None else output

            return tuple_outputs

        dict_outputs = DeformableDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_reference_points=outputs.intermediate_reference_points,
            init_reference_points=outputs.init_reference_points,
            enc_outputs_class=outputs.enc_outputs_class,
            enc_outputs_coord_logits=outputs.enc_outputs_coord_logits,
        )

        return dict_outputs
