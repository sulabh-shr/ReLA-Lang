import logging
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..utils.misc import nested_tensor_from_tensor_list


@torch.jit.script
def refer_ce_loss_jit(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: torch.Tensor):
    loss = F.cross_entropy(inputs, targets, weight=weight)

    return loss


@torch.jit.script
def dice_loss_jit(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: prediction of shape (B, 2, *)
        targets: ground truth of shape (B, 1, *)
                (0 for the negative class and 1 for the positive class).
    """
    inputs = F.softmax(inputs, dim=1)
    inputs = inputs[:, 1, :, :].flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


class ReferringCriterion(nn.Module):
    def __init__(self, weight_dict, losses):
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses

    def get_loss(self, loss, outputs, target_masks, target_nts, weight):
        loss_map = {
            "loss_mask": self.loss_masks,
            "loss_minimap": self.loss_minimap,
            "loss_no_target": self.loss_no_target,
            "loss_dice": self.loss_dice
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, target_masks, target_nts, weight)

    def loss_no_target(self, outputs, target_masks, target_nts, weight):
        src_nt_label = outputs["nt_label"]
        losses = {
            "loss_no_target": refer_ce_loss_jit(src_nt_label, target_nts, weight)
        }
        return losses

    def loss_masks(self, outputs, target_masks, target_nts, weight):
        """ Calculate cross-entropy loss and/or dice loss for masks. """
        src_masks = outputs["pred_masks"]
        h, w = target_masks.shape[-2:]
        src_masks = F.interpolate(src_masks, (h, w), mode='bilinear', align_corners=False)
        target_masks = target_masks.to(src_masks)
        losses = {
            "loss_mask": refer_ce_loss_jit(
                src_masks, target_masks.squeeze(1).long(), weight)
        }
        # Calculate dice loss only if it's coefficient is not 0
        if "loss_dice" in self.weight_dict and self.weight_dict["loss_dice"] != 0:
            losses["loss_dice"] = dice_loss_jit(
                src_masks, target_masks.long())
        return losses

    def loss_dice(self, outputs, target_masks, target_nts, weight):
        """ Calculate dice loss when loss_masks is not used. """
        losses = {}
        # Skip because computed in loss_masks by default
        if "loss_masks" in self.losses:
            return losses
        src_masks = outputs["pred_masks"]
        h, w = target_masks.shape[-2:]
        src_masks = F.interpolate(src_masks, (h, w), mode='bilinear', align_corners=False)
        target_masks = target_masks.to(src_masks)
        losses = {
            "loss_dice": dice_loss_jit(src_masks, target_masks.long())
        }

        return losses

    def loss_minimap(self, outputs, target_masks, target_nts, weight):
        src_minimap = outputs["pred_logits"].permute(0, 2, 1)  # (B, 2, Q)
        queries = src_minimap.shape[-1]
        q_side = np.sqrt(queries)
        assert q_side.is_integer(), f'Query size: {queries} is not a perfect square'
        q_side = int(q_side)

        target_masks = target_masks.to(src_minimap)
        target_minimap = F.interpolate(
            target_masks, (q_side, q_side),
            mode='bilinear', align_corners=False).flatten(start_dim=1)  # (B, 1, Q)
        losses = {
            "loss_minimap": refer_ce_loss_jit(src_minimap, target_minimap.squeeze(1).long(), weight)
        }
        return losses

    def forward(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """ Calculate all losses for main and/or auxiliary predictions.

        Args:
            outputs: model prediction dict with keys:
                pred_masks: n-ary mask of shape (B, nC, H/s, W/s)
                pred_logits: per-query flattened prediction of shape (B, Q, nC)
                nt_label: no-target prediction of shape (B, 2)
                aux_outputs: auxiliary outputs
            targets: Ground truth dict with keys:
                gt_mask_merged: ground truth mask of shape (B, nC, H, W)
        Returns:
            losses: dictionary of losses
        """
        # Pre-merge Ground Truth because all losses require it
        masks = [t["gt_mask_merged"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_nts = torch.stack([t["empty"] for t in targets])  # (B,)
        weight = torch.FloatTensor([0.9, 1.1]).to(outputs["pred_masks"])

        losses = {}

        # Calculate losses for main prediction
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        for loss_name in self.losses:
            if self.weight_dict[loss_name] != 0:
                l_dict = self.get_loss(
                    loss=loss_name,
                    outputs=outputs_without_aux,
                    target_masks=target_masks,
                    target_nts=target_nts,
                    weight=weight
                )
                losses.update(l_dict)

        # Calculate losses for auxiliary prediction
        if 'aux_outputs' in outputs:
            for aux_idx, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss_name in self.losses:
                    if self.weight_dict[loss_name] != 0:
                        l_dict = self.get_loss(
                            loss=loss_name,
                            outputs=aux_outputs,
                            target_masks=target_masks,
                            target_nts=target_nts,
                            weight=weight
                        )
                        l_dict = {f'{k}_{aux_idx}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

        return losses
