from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable

from ..group_vit import GroupingLayer, GroupingBlock
from .referring_transformer_decoder import (
    TRANSFORMER_DECODER_REGISTRY,
    MultiScaleMaskedReferringDecoder,
    CrossAttentionLayer
)


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedLangReferringDecoder(MultiScaleMaskedReferringDecoder):
    @configurable
    def __init__(
            self,
            in_channels,
            mask_classification=True,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,
            mask_dim: int,
            enforce_input_project: bool,
            rla_weight: float = 0.1,
            rla_layers: List[int],
            group_layers: List[int],
            group_tokens: List[int],
            group_out_tokens: List[int],
            group_nheads: List[int],
            group_depths: List[int],
            group_drop_path_rate: int,
            group_hard_assign: bool,
            group_gumbel: bool,
            deep_supervision: bool

    ):
        super().__init__(
            in_channels=in_channels,
            mask_classification=mask_classification,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            mask_dim=mask_dim,
            enforce_input_project=enforce_input_project,
            rla_weight=rla_weight
        )

        # Deep supervision similar to MaskFormer
        self.deep_supervision = deep_supervision
        self.rla_layers = rla_layers
        self.RLA_lang_att = nn.ModuleList()
        self.LangGroupLayers = nn.ModuleList()
        self.group_layers = group_layers

        assert all([i < self.num_layers for i in self.group_layers]), \
            f'Group layers: {self.group_layers} exceeds number of layers: {self.num_layers}'

        dpr = [x.item() for x in torch.linspace(0, group_drop_path_rate, sum(group_depths))]

        group_idx = 0
        for i_layer in range(self.num_layers):
            if i_layer in self.rla_layers:
                self.RLA_lang_att.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm
                    )
                )
            if i_layer in self.group_layers:
                downsample = GroupingBlock(
                    dim=hidden_dim,
                    out_dim=hidden_dim,
                    num_heads=group_nheads[group_idx],
                    num_group_token=group_tokens[group_idx],
                    num_output_group=group_out_tokens[group_idx],
                    norm_layer=nn.LayerNorm,
                    hard=group_hard_assign,
                    gumbel=group_gumbel)
                self.LangGroupLayers.append(
                    GroupingLayer(
                        dim=hidden_dim,
                        num_input_token=20,
                        depth=group_depths[group_idx],
                        num_heads=group_nheads[group_idx],
                        num_group_token=group_tokens[group_idx],
                        mlp_ratio=4.,
                        drop_path=dpr[sum(group_depths[:group_idx]):sum(group_depths[:group_idx + 1])],
                        norm_layer=nn.LayerNorm,
                        downsample=downsample,
                        use_checkpoint=False,
                        group_projector=None,
                        zero_init_group_token=False
                    )
                )
                group_idx += 1

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = 1
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        # RLA layers
        ret["rla_layers"] = cfg.MODEL.MASK_FORMER.RLA_LAYERS

        # Grouping Parameters
        ret["group_layers"] = cfg.MODEL.MASK_FORMER.GROUP_LAYERS
        ret["group_tokens"] = cfg.MODEL.MASK_FORMER.GROUP_TOKENS
        ret["group_out_tokens"] = cfg.MODEL.MASK_FORMER.GROUP_OUT_TOKENS
        ret["group_nheads"] = cfg.MODEL.MASK_FORMER.GROUP_NHEADS
        ret["group_depths"] = cfg.MODEL.MASK_FORMER.GROUP_DEPTHS
        ret["group_drop_path_rate"] = cfg.MODEL.MASK_FORMER.GROUP_DROP_PATH_RATE
        ret["group_hard_assign"] = cfg.MODEL.MASK_FORMER.GROUP_HARD_ASSIGN
        ret["group_gumbel"] = cfg.MODEL.MASK_FORMER.GROUP_GUMBEL
        ret["deep_supervision"] = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION

        return ret

    def forward(
            self,
            x: List[torch.Tensor],
            mask_features: torch.Tensor,
            lang_feat: torch.Tensor,
            mask=None
    ):
        """

        Args:
            x: List of multiscale features in top-down order
                i.e. [..., layer_n-1, layer_n] of shapes
                [..., (B, C_conv, H/16, W/ 16), (B, C_conv, H/8, W/8)]
            mask_features: Mask features of shape (B, C_mask, H/4, W/4)
            lang_feat: Language feature of shape (B, C_l, N_l)
            mask:

        Returns:
            'pred_logits': (B, Q, 2)
            'pred_masks': (B, 2, H/4, W/4)
            'all_masks': outputs_mask,  # Not used anywhere
            'nt_label': No-target label of shape
        """

        # print('-' * 70)
        # print('MultiScaleMaskedLangReferringDecoder')

        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []  # List of shape of features from bottom to top stages

        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # Learned Positional Embedding
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # (Q, B, C)
        # Learned Queries
        prev_query_output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)  # (Q, B, C)

        predictions_class = []
        # predictions_mask = []

        aux_tgt_mask = None
        aux_nt_label = None
        if self.deep_supervision:
            aux_tgt_mask = []
            aux_nt_label = []

        # prediction heads on learnable query features
        outputs_minimap, outputs_mask, attn_mask, tgt_mask, nt_label = self.forward_prediction_heads(
            prev_query_output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_minimap)
        # predictions_mask.append(outputs_mask)

        # Project language dim (C_l) to vision dim (C)
        lang_feat_att = lang_feat.permute(0, 2, 1)  # (B, N_l, C_l)
        lang_feat_att = self.lang_proj(lang_feat_att)  # (B, N_l, C)

        # ReLA is applied multiple times for performance
        group_idx = rla_idx = 0
        prev_group_token = None
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # Queries with all location masks skipped are inverted to attend to all locations
            all_masked_locations = torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])
            attn_mask[all_masked_locations] = False  # (B*nHeads, Q, Hi * Wi)
            # cross-attention of regions and vision features
            prev_query_output = self.RIA_layers[i](
                tgt=prev_query_output,  # query
                memory=src[level_index],  # key
                memory_mask=attn_mask,  # attn_mask
                memory_key_padding_mask=None,
                pos=pos[level_index],  # added to key/memory
                query_pos=query_embed  # added to query/tgt
            )

            # Language Grouping before RLA
            if i in self.group_layers:
                grouping_layer = self.LangGroupLayers[group_idx]
                lang_feat_att, prev_group_token, attn_dict = grouping_layer(
                    x=lang_feat_att,  # [B, N_l, C]
                    prev_group_token=prev_group_token,  # [B, S_1, C]
                    return_attn=False if self.training else True
                )
                group_idx += 1

            # Region-Language Cross-Attention
            if i in self.rla_layers:
                lang_vision_feat = (
                        self.RLA_lang_att[rla_idx](
                            prev_query_output,  # (Q, B, C)
                            lang_feat_att.permute(1, 0, 2)  # (N_l, B, C)
                        ) *
                        F.sigmoid(self.lang_weight)  # (1,)
                )  # (Q, B, C)
                prev_query_output = prev_query_output + lang_vision_feat * self.rla_weight  # (Q, B, C)
                rla_idx += 1

            # RLA vision attention
            # self attention itself has a skip connection
            prev_query_output = self.RLA_vision[i](  # Self-Attention
                prev_query_output,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # Postprocessing
            prev_query_output = self.transformer_ffn_layers[i](prev_query_output)

            outputs_minimap, outputs_mask, attn_mask, tgt_mask, nt_label = (
                self.forward_prediction_heads(
                    prev_query_output,
                    mask_features,
                    attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]))

            # Predictions of all passes are recorded, but only the last output is used in this code
            predictions_class.append(outputs_minimap)
            # predictions_mask.append(outputs_mask)

            if self.deep_supervision:
                aux_tgt_mask.append(tgt_mask)
                aux_nt_label.append(nt_label)

        out = {
            'pred_logits': predictions_class[-1],  # (B, Q, nC)
            'pred_masks': tgt_mask,  # (B, nC, H/4, W/4)
            'all_masks': outputs_mask,  # Not used anywhere
            'nt_label': nt_label
        }

        if self.deep_supervision:
            aux_outputs = []
            for i in range(self.num_layers):
                layer_output = {
                    'pred_logits': predictions_class[i + 1],  # skip first prediction
                    'pred_masks': aux_tgt_mask[i],
                    'nt_label': aux_nt_label[i],

                }
                aux_outputs.append(layer_output)
            out['aux_outputs'] = aux_outputs

        return out
