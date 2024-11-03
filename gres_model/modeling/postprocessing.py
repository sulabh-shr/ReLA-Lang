from torch.nn import functional as F

from ..utils.misc import get_pad_values


def refer_postprocess(result, img_size, output_height, output_width):
    """
    Return mask predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    max_size = result.shape[1:]
    left_p, right_p, top_p, bottom_p = get_pad_values(max_size, img_size)
    right_idx = max_size[-1] - right_p
    bottom_idx = max_size[-2] - bottom_p
    result = result[:, top_p:bottom_idx, left_p: right_idx].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result
