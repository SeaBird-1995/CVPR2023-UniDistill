'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-04 00:06:09
Email: haimingzhang@link.cuhk.edu.cn
Description: The nuscenes dataset for occupancy prediction task.
'''

from tqdm import tqdm
import numpy as np
import os

from .nuscenes_multimodal import NuscenesMultiModalData
from .occ_metrics import Metric_mIoU, Metric_FScore


def load_occ_gt_from_file(results: dict):
    occ_gt_path = results['occ_gt_path']
    occ_gt_path = os.path.join(occ_gt_path, "labels.npz")

    occ_labels = np.load(occ_gt_path)
    semantics = occ_labels['semantics']
    mask_lidar = occ_labels['mask_lidar']
    mask_camera = occ_labels['mask_camera']

    results['voxel_semantics'] = semantics
    results['mask_lidar'] = mask_lidar
    results['mask_camera'] = mask_camera

    return results


class NuScenesDatasetOccpancy(NuscenesMultiModalData):
    def evaluate(self, occ_results, **eval_kwargs):
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=True)

        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.infos[index]

            occ_gt = np.load(os.path.join(info['occ_path'],'labels.npz'))
            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)
            # occ_pred = occ_pred
            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)

        return self.occ_eval_metrics.count_miou()
    
    def __len__(self):
        return 100
    
    def __getitem__(self, idx):
        data_dict = super(NuScenesDatasetOccpancy, self).__getitem__(idx)
        data_dict['occ_gt_path'] = self.infos[idx]['occ_path']

        data_dict = load_occ_gt_from_file(data_dict)
        return data_dict

