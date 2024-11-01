import itertools
import os
import cv2
import json
import tqdm
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import product

from gres_model.data.datasets.refer import REFER
from gres_model.data.datasets.grefer import G_REFER

img_root = 'D:/workspace/datasets/coco/images'
data_root = 'D:/workspace/datasets/coco/'
json_path = 'D:/workspace/datasets/coco/grefcoco/grefs(unc).json'
save_path = 'D:/workspace/datasets/coco/grefcoco/grefs(gmu).json'

SELECTED = [
    'GRES'
]
datasets = {
    'RefCOCO-G':
        {
            'referAnn': 'D:/workspace/datasets/coco/refcoco/refs(google).p',
            'instance': 'D:/workspace/datasets/coco/refcoco/instances.json',
            'name': 'refcoco',
            'splitBy': 'google'
        },
    'RefCOCO-U':
        {
            'referAnn': 'D:/workspace/datasets/coco/refcoco/refs(unc).p',
            'instance': 'D:/workspace/datasets/coco/refcoco/instances.json',
            'name': 'refcoco',
            'splitBy': 'unc'
        },
    'GRES':
        {
            'referAnn': 'D:/workspace/datasets/coco/grefcoco/grefs(unc).json',
            'instance': 'D:/workspace/datasets/coco/grefcoco/instances.json',
            'name': 'grefcoco',
            'splitBy': 'unc'
        },
    'GoogleRef':
        {
            'referAnn': 'D:/workspace/datasets/coco/refcocog/refs(google).p',
            'instance': 'D:/workspace/datasets/coco/refcocog/instances.json',
            'name': 'refcocog',
            'splitBy': 'google'
        },
    'RefCOCO+':
        {
            'referAnn': 'D:/workspace/datasets/coco/refcoco+/refs(unc).p',
            'instance': 'D:/workspace/datasets/coco/refcoco+/instances.json',
            'name': 'refcoco+',
            'splitBy': 'google'
        },

}

data_objects = {

}

for data_key in SELECTED:
    dataset_info = datasets[data_key]
    if dataset_info['name'] == 'grefcoco':
        data_obj = G_REFER(
            data_root=data_root,
            dataset='grefcoco',
            splitBy='unc'
        )
    else:
        data_obj = REFER(
            data_root=data_root,
            dataset=dataset_info['name'],
            splitBy=dataset_info['splitBy']
        )
    data_objects[data_key] = data_obj

# ------------------------------------------------------------------------------

with open(json_path) as f:
    json_content = json.load(f)

image_splits = defaultdict(set)
new_content = {}
# ------------------------------------------------------------------------------

gres = data_objects['GRES']

for ref_dict in tqdm.tqdm(json_content):

    ref_id = ref_dict['ref_id']
    ref = gres.loadRefs(ref_id)[0]

    refAnnIds = ref['ann_id']
    refCatIds = ref['category_id']

    num_ref = len(refAnnIds)

    # Skip no-target
    if -1 in refAnnIds:
        continue

    ref_dict['referents'] = num_ref

    imgAnnIds = gres.getAnnIds(image_ids=ref['image_id'])
    imgAnns = gres.loadAnns(ann_ids=imgAnnIds)

    distractor_ann_ids = []
    distractor_cat_ids = []
    other_ann_ids = []
    other_cat_ids = []

    for img_ann in imgAnns:
        imgAnnId = img_ann['id']
        imgCatId = img_ann['category_id']
        if imgAnnId not in refAnnIds:
            if imgCatId in refCatIds:
                distractor_ann_ids.append(imgAnnId)
                distractor_cat_ids.append(imgCatId)
            else:
                other_ann_ids.append(imgAnnId)
                other_cat_ids.append(imgCatId)

    ref_dict['ann_id'] = ref_dict['ann_id'] + distractor_ann_ids + other_ann_ids
    ref_dict['category_id'] = ref_dict['category_id'] + distractor_cat_ids + other_cat_ids
    ref_dict['distractors'] = len(distractor_ann_ids)
    ref_dict['non_distractors'] = len(other_ann_ids)

with open(save_path, 'w') as f:
    json.dump(json_content, f)

# ------------------------------------------------------------------------------
