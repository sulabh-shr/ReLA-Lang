from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import ReferringCriterion
from .modeling.postprocessing import refer_postprocess
from .structures import ImageList
from .utils.misc import get_pad_values
from .utils.tokens import get_tokenizer


@META_ARCH_REGISTRY.register()
class GRES(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        lang_backbone: nn.Module,
        pad_value: int = 0,
        label_pad_value: int = 255,
    ):

        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.pad_value = pad_value
        self.label_pad_value = label_pad_value
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # language backbone
        self.text_encoder = lang_backbone

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Setup text encoder and freeze layers
        text_encoder = get_tokenizer(cfg.REFERRING.BERT_TYPE)
        if not isinstance(text_encoder, str):
            text_encoder.pooler = None

            # Freeze except last layers
            freeze_at = cfg.REFERRING.get("FREEZE_AT", 0)
            if freeze_at > 0:
                print(f"Freezing BERT layers [0,{freeze_at})")
                for name, param in text_encoder.named_parameters():
                    param.requires_grad = False
                    if "encoder.layer" in name:
                        encoder_layer_num = int(name.split(".")[2])
                        if encoder_layer_num >= freeze_at:
                            param.requires_grad = True

        # Loss weights
        weight_dict = {
            "loss_mask": cfg.MODEL.MASK_FORMER.MASK_WEIGHT,
            "loss_dice": cfg.MODEL.MASK_FORMER.DICE_WEIGHT,
            "loss_minimap": cfg.MODEL.MASK_FORMER.MINIMAP_WEIGHT,
            "loss_no_target": cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT,
            "loss_attn": cfg.MODEL.MASK_FORMER.ATTN_LOSS_WEIGHT,
            "loss_distractor": cfg.MODEL.MASK_FORMER.DISTRACTOR_WEIGHT,
        }
        weight_dict = {k: v for k, v in weight_dict.items() if v != 0}
        losses = [k for k in weight_dict]
        if "loss_distractor" in weight_dict:
            assert cfg.INPUT.USE_DISTRACTORS, f"Set INPUT.USE_DISTRACTORS to True for loss_distractor"

        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            aux_weight_multiplier = torch.linspace(0.1, 0.9, dec_layers - 2).tolist()
            for aux_idx in range(dec_layers - 2):
                aux_weight_dict.update(
                    {
                        f"{k}_{aux_idx}": v * aux_weight_multiplier[aux_idx]
                        for k, v in weight_dict.items()
                        if k != "loss_attn"
                    }
                )
            weight_dict.update(aux_weight_dict)

        criterion = ReferringCriterion(
            weight_dict=weight_dict,
            losses=losses,
            ignore_index=cfg.INPUT.LABEL_PAD_VALUE,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "lang_backbone": text_encoder,
            "pad_value": cfg.INPUT.PAD_VALUE,
            "label_pad_value": cfg.INPUT.LABEL_PAD_VALUE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs: List):

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility, self.pad_value)

        lang_tokens = [x["lang_tokens"].to(self.device) for x in batched_inputs]
        lang_tokens = torch.cat(lang_tokens, dim=0)

        lang_mask = torch.cat([x["lang_mask"].to(self.device) for x in batched_inputs], dim=0)

        # When using saved embeddings, language embeddings are loaded as tokens
        lang_feat = lang_tokens
        if not isinstance(self.text_encoder, str):
            lang_feat = self.text_encoder(lang_tokens, attention_mask=lang_mask)[0]  # (B, Nl, 768)

        lang_feat = lang_feat.permute(0, 2, 1)  # (B, 768, N_l)
        lang_mask = lang_mask.unsqueeze(dim=-1)  # (B, 768, N_l, 1)

        features = self.backbone(images.tensor, lang_feat, lang_mask)
        outputs = self.sem_seg_head(features, lang_feat, lang_mask)

        if self.training:
            targets = self.prepare_targets(batched_inputs, images)
            losses = self.criterion(outputs, targets)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
            return losses
        else:
            mask_pred_results = outputs["pred_masks"]
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            nt_pred_results = outputs["nt_label"]

            batch_attn = outputs.get("attn", None)

            del outputs

            processed_results: List[Dict] = []
            for batch_idx, mask_pred_result, nt_pred_result, input_per_image, image_size in zip(
                range(len(batched_inputs)),
                mask_pred_results,
                nt_pred_results,
                batched_inputs,
                images.image_sizes,
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                mask_pred_result = retry_if_cuda_oom(refer_postprocess)(mask_pred_result, image_size, height, width)

                r, nt = retry_if_cuda_oom(self.refer_inference)(mask_pred_result, nt_pred_result)
                processed_results[-1]["ref_seg"] = r
                processed_results[-1]["nt_label"] = nt

                if batch_attn is not None:
                    for attn_type in ["soft", "hard"]:
                        img_attn = {}
                        for layer, layer_attn in batch_attn.items():
                            img_attn[layer] = layer_attn[attn_type][batch_idx][0]
                        processed_results[-1][f"{attn_type}_attn"] = img_attn

            return processed_results

    def prepare_targets(self, batched_inputs: List, images: ImageList) -> List[Dict[str, torch.Tensor]]:
        """

        ImageList pads the input image tensor to be size_divisible.
        Here, the ground truth masks are padded in the same way.

        Args:
            batched_inputs: original input list
            images: padded image tensor

        Returns:
            new_targets: padded gt label masks
        """

        new_targets = []
        max_size = images.tensor.shape[-2:]

        for data_per_image in batched_inputs:

            targets_per_image = data_per_image["instances"]
            target_dict = {
                "empty": torch.tensor(
                    data_per_image["empty"], dtype=targets_per_image.gt_classes.dtype, device=self.device
                )
            }

            for key in (
                "gt_mask_merged",
                "distractors_merged",
                "non_distractors_merged",
            ):
                if key not in data_per_image:
                    continue
                mask = data_per_image[key]
                mask_size = mask.shape  # (1, H, W)

                left_p, right_p, top_p, bottom_p = get_pad_values(max_size, mask_size[1:])
                padding_size = [left_p, right_p, top_p, bottom_p]
                new_mask = F.pad(mask, padding_size, value=self.label_pad_value)
                target_dict[f"{key}-resized"] = new_mask

            new_targets.append(target_dict)

        return new_targets

    def refer_inference(self, mask_pred, nt_pred):
        mask_pred = mask_pred.softmax(dim=0)
        nt_pred = nt_pred.softmax(dim=0)
        return mask_pred, nt_pred
