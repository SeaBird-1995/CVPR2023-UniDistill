'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-03 20:41:53
Email: haimingzhang@link.cuhk.edu.cn
Description: The lidar only branch for Occupancy prediction.
'''

from unidistill.exps.base_cli import run_cli
from unidistill.exps.multisensor_fusion.nuscenes.Occupancy.occupancy_centerhead_fusion_exp import (
    Exp as BaseExp,
)
from unidistill.exps.multisensor_fusion.nuscenes._base_.base_occ_nuscenes_cfg import (
    CENTERPOINT_DET_HEAD_CFG,
    MODEL_CFG,
    DATA_CFG
)


class Exp(BaseExp):
    def __init__(
        self,
        batch_size_per_device=4,
        total_devices=1,
        max_epochs=20,
        ckpt_path=None,
        **kwargs
    ):
        custom_model_cfg = MODEL_CFG
        custom_model_cfg["camera_encoder"] = None
        custom_data_cfg = DATA_CFG
        custom_data_cfg["img_key_list"] = []
        
        super(Exp, self).__init__(
            batch_size_per_device, total_devices, max_epochs, ckpt_path, 
            custom_data_cfg, custom_model_cfg)


if __name__ == "__main__":
    import logging

    logging.getLogger("mmcv").disabled = True
    logging.getLogger("mmseg").disabled = True
    run_cli(Exp, "BEVFusion_nuscenes_centerhead_lidar_exp")
