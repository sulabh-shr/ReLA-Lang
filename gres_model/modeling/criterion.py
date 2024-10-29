import logging
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size

from ..utils.misc import nested_tensor_from_tensor_list, is_dist_avail_and_initialized


@torch.jit.script
def ce_loss_jit(inputs: torch.Tensor, targets: torch.Tensor,
                weight: torch.Tensor = None) -> torch.Tensor:
    """ Cross-entropy loss

    Args:
        inputs: predictions of shape (B, nC, *)
        targets: targets of shape (B, *)
        weight: weights of shape (nC)

    Returns:
        loss: mean cross-entropy loss
    """
    loss = F.cross_entropy(inputs, targets, weight=weight)
    return loss


@torch.jit.script
def dice_loss_jit(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """ Compute the DICE loss, similar to generalized IOU for masks

    Args:
        inputs: prediction of shape (B, 2, H, W)
        targets: ground truth of shape (B, H, W)
                (0 for the negative class and 1 for the positive class).

    Returns:
        loss: mean dice loss
    """
    inputs = F.softmax(inputs, dim=1)
    inputs = inputs[:, 1, :, :].flatten(1)  # take 1 for presence of class
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


@torch.jit.script
def softmax_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """ Focal loss for logits with 2 dimension.

    Args:
        inputs: predictions of shape (B, 2, *)
        targets: targets of shape (B, *)
        alpha: (optional) weighting factor in range (0,1) to balance
        gamma: exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    inputs = F.softmax(inputs, dim=1)

    # Gather the probabilities of the true class for each pixel
    # inputs: [N, C, H, W] -> [N, H, W]
    probs = inputs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Compute the focal loss based on the probability of the true class
    loss = - alpha * (1 - probs) ** gamma * torch.log(probs + 1e-6)

    return loss.mean()


class ReferringCriterion(nn.Module):
    def __init__(self, weight_dict, losses):
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses

    @staticmethod
    def _get_query_side(queries: torch.Tensor) -> int:
        """ Get side length of minimap from flattened queries.

        Args:
            queries: minimap of shape (B, nC, Q) where Q is a whole square

        Returns:
            q_side: side length of minimap assuming a whole square
        """
        num_queries = queries.shape[-1]
        q_side = np.sqrt(num_queries)
        assert q_side.is_integer(), f'Query size: {num_queries} is not a perfect square'
        q_side = int(q_side)
        return q_side

    def get_loss(self, loss: str, outputs: Dict, targets: Dict):
        """ Map of possible losses and function to calculate them. """
        loss_map = {
            "loss_mask": self.loss_masks,
            "loss_minimap": self.loss_minimap,
            "loss_no_target": self.loss_no_target,
            "loss_dice": self.loss_dice,
            "loss_attn": self.loss_attn
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets)

    def loss_no_target(self, outputs: Dict, targets: Dict):
        """ Loss for target vs no-target binary classification """
        src_nt_label = outputs["nt_label"]
        target_nts = targets['target_nts']
        binary_weight = targets['weight']
        losses = {
            "loss_no_target": ce_loss_jit(src_nt_label, target_nts, binary_weight)
        }
        return losses

    def loss_masks(self, outputs: Dict, targets: Dict):
        """ Calculate cross-entropy loss and/or dice loss for masks. """

        target_masks = targets['target_masks_int']  # (B, H, W)
        h, w = target_masks.shape[-2:]

        pred_masks = outputs["pred_masks"]
        pred_masks = F.interpolate(pred_masks, (h, w), mode='bilinear', align_corners=False)

        binary_weight = targets['weight']

        losses = {
            "loss_mask": ce_loss_jit(pred_masks, target_masks, binary_weight)
        }

        # Calculate dice loss only if it's coefficient is not 0
        if "loss_dice" in self.weight_dict and self.weight_dict["loss_dice"] != 0:
            losses["loss_dice"] = dice_loss_jit(
                pred_masks, target_masks)

        return losses

    def loss_dice(self, outputs: Dict, targets: Dict):
        """ Calculate dice loss when loss_masks is not used. """

        losses = {}

        # Skip because computed in loss_masks by default
        if "loss_masks" in self.losses:
            return losses

        target_masks = targets['target_masks_int']  # (B, H, W)
        pred_masks = outputs["pred_masks"]
        h, w = target_masks.shape[-2:]
        pred_masks = F.interpolate(pred_masks, (h, w), mode='bilinear', align_corners=False)

        losses = {
            "loss_dice": dice_loss_jit(pred_masks, target_masks)
        }

        return losses

    def loss_minimap(self, outputs: Dict, targets: Dict):
        """ Calculate loss for queries minimap prediction. """

        binary_weight = targets['weight']
        target_minimap = targets['target_minimap_int']  # (B, Q)
        pred_minimap = outputs["pred_logits"].permute(0, 2, 1)  # (B, nC, Q)

        if target_minimap.shape[-1] != pred_minimap.shape[-1]:
            target_masks = targets['target_masks']
            q_side = self._get_query_side(pred_minimap)
            target_minimap = F.interpolate(
                target_masks, (q_side, q_side),
                mode='bilinear', align_corners=False).flatten(start_dim=1)  # (B, 1, Q)
            target_minimap = target_minimap.squeeze(1).long()  # (B, Q)

        losses = {
            "loss_minimap": ce_loss_jit(pred_minimap, target_minimap, binary_weight)
        }

        return losses

    def loss_attn(self, outputs, *args, **kwargs):
        """ Attention regularization """

        attn_per_stage = outputs['attn']
        max_stage = max(attn_per_stage)
        stage_weights = np.arange(max_stage + 1, 0, -1)
        stage_weights = stage_weights / np.sum(stage_weights)

        losses = {"loss_attn": 0}

        for stage, stage_attn in attn_per_stage.items():
            attn: torch.Tensor = stage_attn['attn']  # (B, 1, Go, Gi)
            b, _, go, gi = attn.shape

            # max input assigned to each output group
            group_base = gi / go * 1.5
            group_wt_sum = torch.sum(attn, dim=-1)  # (B, 1, Go)
            diff_from_base = group_base - group_wt_sum
            diff_from_base = torch.clamp(diff_from_base, min=0)
            # group_loss = torch.mean(torch.pow(diff_from_base, 2))
            group_loss = torch.mean(diff_from_base)

            # entropy loss per input group
            attn_non_zero = torch.clamp(attn, min=1e-12)
            input_entropy = -torch.sum(attn * torch.log(attn_non_zero), dim=-2)  # (B, 1, Gi)
            entropy_loss = torch.mean(input_entropy)

            losses["loss_attn"] += (group_loss + entropy_loss) * stage_weights[stage]

        losses["loss_attn"] = losses["loss_attn"] / len(attn_per_stage)

        return losses

    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """ Calculate all losses for main and/or auxiliary predictions.

        Args:
            outputs: model prediction dict with keys:
                pred_masks: n-ary mask of shape (B, nC, H/s, W/s)
                pred_logits: per-query flattened prediction of shape (B, Q, nC)
                nt_label: no-target prediction of shape (B, 2)
                aux_outputs: list of auxiliary outputs with same format as output
                attn: attention per grouping stage of shape (B, 1, Go, Gi)
            targets: list of ground truth dict with keys:
                gt_mask_merged: ground truth mask of shape (B, nC, H, W)
                empty: ground truth no-target label
        Returns:
            losses: dictionary of losses
        """

        # Pre-compute gt related info because all losses require it
        masks = [t["gt_mask_merged"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(outputs['pred_masks'])
        target_nts = torch.stack([t["empty"] for t in targets])

        # Pre-compute the minimap target
        # This is especially helpful with aux-output
        pred_minimap = outputs['pred_logits']  # (B, Q, 2)
        q_side = self._get_query_side(pred_minimap.permute(0, 2, 1))
        target_minimap = F.interpolate(
            target_masks, (q_side, q_side),
            mode='bilinear', align_corners=False).flatten(start_dim=1)  # (B, 1, Q)
        target_minimap = target_minimap.squeeze(1).long()

        # Weight for no-target vs target class
        weight = torch.FloatTensor([1.0, 1.0]).to(outputs["pred_masks"])

        targets = {
            'target_masks': target_masks,  # (B, 1, H, W)
            'target_masks_int': target_masks.squeeze(1).long(),  # (B, H, W)
            'target_nts': target_nts,  # (B,)
            'target_minimap_int': target_minimap,  # (B, Q)
            'weight': weight,
        }

        losses = {}

        # Calculate losses for main prediction
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        for loss_name in self.losses:
            if self.weight_dict[loss_name] != 0:
                l_dict = self.get_loss(loss=loss_name, targets=targets,
                                       outputs=outputs_without_aux)
                losses.update(l_dict)

        # Calculate losses for auxiliary prediction
        if 'aux_outputs' in outputs:
            for aux_idx, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss_name in self.losses:
                    # attention loss only calculated once in main output
                    if loss_name == 'loss_attn':
                        continue
                    if self.weight_dict[loss_name] != 0:
                        l_dict = self.get_loss(loss=loss_name, targets=targets,
                                               outputs=aux_outputs)
                        l_dict = {f'{k}_{aux_idx}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

        return losses
