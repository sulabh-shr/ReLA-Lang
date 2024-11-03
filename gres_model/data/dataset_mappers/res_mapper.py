import copy
import logging
from typing import Dict

import torch
import numpy as np

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import Instances, PolygonMasks

from transformers import BertTokenizer
from pycocotools import mask as coco_mask

__all__ = ["RefCOCOMapperV2"]


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_train(cfg):
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MIN_SCALE

    augmentation = [
        T.ResizeShortestEdge(image_size, image_size),
        T.RandomApply(T.ResizeScale(min_scale, max_scale, image_size, image_size), prob=0.5),
        T.RandomApply(T.Resize((image_size, image_size)), prob=0.25),
        T.RandomApply(T.RandomBrightness(0.25, 1.25), prob=0.5),
        T.RandomApply(T.RandomContrast(0.5, 1.5), prob=0.5),
    ]

    return augmentation


def build_transform_test(cfg):
    image_size = cfg.INPUT.IMAGE_SIZE

    augmentation = [
        T.ResizeShortestEdge(image_size, image_size),
    ]

    return augmentation


# This is specifically designed for the COCO dataset.
class RefCOCOMapperV2:
    @configurable
    def __init__(
            self,
            is_train=True,
            *,
            tfm_gens,
            image_format,
            bert_type,
            max_tokens,
            merge=True,
            label_pad_value=255,
            use_distractors=False
    ):
        self.is_train = is_train
        self.merge = merge
        self.label_pad_value = label_pad_value
        self.use_distractors = use_distractors
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "Full TransformGens used: {}".format(str(self.tfm_gens))
        )

        self.bert_type = bert_type
        self.max_tokens = max_tokens
        logging.getLogger(__name__).info(
            "Loading BERT tokenizer: {}...".format(self.bert_type)
        )
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_type)

        self.img_format = image_format

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        if is_train:
            tfm_gens = build_transform_train(cfg)
        else:
            tfm_gens = build_transform_test(cfg)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "bert_type": cfg.REFERRING.BERT_TYPE,
            "max_tokens": cfg.REFERRING.MAX_TOKENS,
            "label_pad_value": cfg.INPUT.LABEL_PAD_VALUE,
            "use_distractors": cfg.INPUT.USE_DISTRACTORS
        }
        return ret

    @staticmethod
    def _merge_masks(x):
        return x.sum(dim=0, keepdim=True).clamp(max=1)

    def __call__(self, dataset_dict: Dict) -> Dict:
        """ Load and convert dataset_dict for model.

        If distractors and non_distractors exist in annotations, they are first
        transformed together with segment mask and separated later.

        Args:
            dataset_dict: Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dataset_dict with images and ground truths:
                image: contains transformed images of shape (3, H, W)
                empty: boolean flag for no-target
                instances: detectron2 instances with individual gt_masks and gt_boxes
                gt_mask_merged: combined binary mask for target of shape (1, H, W)
                lang_tokens: language model tokens of shape (1, N_l)
                lang_mask: binary mask of shape (1, N_l) with
                            1 for valid input tokens
                            0 for invalid input tokens
        """

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        org_image_size = image.shape[:2]

        # Transform the image
        aug_input = T.AugInput(image)
        image, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # For testing, do not reshape the ground truth masks
        image_size = image.shape[:2]
        if not self.is_train:
            image_size = org_image_size
            transforms = []

        # Load annotations only for images with referent target and transform them as segment points
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_size)
            for obj in dataset_dict.pop("annotations")
            if (obj.get("iscrowd", 0) == 0) and (not obj.get("empty", False))
        ]
        # Convert points to masks and store them in instances
        instances: Instances = utils.annotations_to_instances(annos, image_size)

        empty = dataset_dict.get("empty", False)

        if len(instances) > 0:
            assert (not empty)
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Generate masks from polygon
            h, w = instances.image_size
            assert hasattr(instances, 'gt_masks')
            gt_masks = instances.gt_masks
            gt_masks = convert_coco_poly_to_mask(gt_masks, h, w)
            instances.gt_masks = gt_masks

            # Separate referents, distractors and non-distractors
            if self.use_distractors:
                num_gt = dataset_dict.pop("referents", 0)
                num_distractors = dataset_dict.pop("distractors", 0)
                num_non_distractors = dataset_dict.pop("non_distractors", 0)
                if self.use_distractors:
                    distractors = instances[num_gt:num_gt + num_distractors]
                    distractors_masks = distractors.gt_masks
                    non_distractors = instances[num_gt + num_distractors:num_gt + num_distractors + num_non_distractors]
                    non_distractors_masks = non_distractors.gt_masks
                instances = instances[:num_gt]
                gt_masks = gt_masks[:num_gt]
        else:
            assert empty
            h, w = image_size
            gt_masks = torch.zeros((0, h, w), dtype=torch.uint8)
            instances.gt_masks = gt_masks
            if self.use_distractors:
                distractors_masks = torch.zeros((0, h, w), dtype=torch.uint8)
                non_distractors_masks = torch.zeros((0, h, w), dtype=torch.uint8)

        if self.is_train:
            dataset_dict["instances"] = instances
        else:
            dataset_dict["gt_mask"] = gt_masks

        dataset_dict["empty"] = empty

        if self.merge:
            dataset_dict["gt_mask_merged"] = self._merge_masks(gt_masks)
            if self.use_distractors:
                distractors_masks = self._merge_masks(distractors_masks)
                non_distractors_masks = self._merge_masks(non_distractors_masks)
                # Set distractors as background and other pixels as ignore class
                distractors_merged = torch.ones_like(distractors_masks) * self.label_pad_value
                non_distractors_merged = torch.ones_like(non_distractors_masks) * self.label_pad_value
                distractors_merged[distractors_masks == 1] = 0
                non_distractors_merged[non_distractors_masks == 1] = 0
                dataset_dict['distractors_merged'] = distractors_merged  # (1, H, W)
                dataset_dict['non_distractors_merged'] = non_distractors_merged  # (1, H, W)

        # Language data
        sentence_raw = dataset_dict['sentence']['raw']
        attention_mask = [0] * self.max_tokens
        padded_input_ids = [0] * self.max_tokens

        input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

        input_ids = input_ids[:self.max_tokens]
        padded_input_ids[:len(input_ids)] = input_ids

        attention_mask[:len(input_ids)] = [1] * len(input_ids)

        dataset_dict['lang_tokens'] = torch.tensor(padded_input_ids).unsqueeze(0)
        dataset_dict['lang_mask'] = torch.tensor(attention_mask).unsqueeze(0)

        return dataset_dict
