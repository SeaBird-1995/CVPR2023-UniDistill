
from typing import Any

import mmcv
from mmcv.cnn.bricks.conv_module import ConvModule
import torch
from torch import Tensor
import torch.nn as nn
from einops import rearrange
import numpy as np
from functools import partial

from unidistill.exps.base_cli import run_cli
from unidistill.exps.multisensor_fusion.nuscenes._base_.base_occ_nuscenes_cfg import (
    CENTERPOINT_DET_HEAD_CFG,
    DATA_CFG,
    MODEL_CFG
)
from unidistill.data.multisensorfusion.nuscenes_multimodal import (
    collate_fn,
)
from unidistill.exps.multisensor_fusion.nuscenes.BEVFusion.BEVFusion_nuscenes_base_exp import (
    BEVFusion,
)
from unidistill.utils import torch_dist
from unidistill.exps.multisensor_fusion.nuscenes.BEVFusion.BEVFusion_nuscenes_base_exp import (
    Exp as BaseExp,
)
from unidistill.data.multisensorfusion.nuscenes_multimodal_occ import (
    NuScenesDatasetOccpancy,
)

from mmdet.models.losses import CrossEntropyLoss

_IMG_BACKBONE_CONF = dict(
    type="ResNet",
    depth=50,
    frozen_stages=0,
    out_indices=[0, 1, 2, 3],
    norm_eval=False,
    init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
)


_IMG_NECK_CONF = dict(
    type="SECONDFPN",
    in_channels=[256, 512, 1024, 2048],
    upsample_strides=[0.25, 0.5, 1, 2],
    out_channels=[128, 128, 128, 128],
)

_DEPTH_NET_CONF = dict(in_channels=512, mid_channels=512)


class OccupancyHead(nn.Module):
    def __init__(self, det_head_cfg: mmcv.Config, **kwargs):
        super().__init__()
        self.build_occ_head(det_head_cfg)
        self.use_mask = True
        self.loss_occ = CrossEntropyLoss(use_sigmoid=False, loss_weight=1.0)

    def build_occ_head(self, head_cfg):
        in_channels = head_cfg.in_channels
        out_channels = head_cfg.out_channels
        num_classes = head_cfg.num_classes
        self.num_classes = num_classes

        self.final_conv = ConvModule(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))
        self.predicter = nn.Sequential(
                nn.Linear(out_channels, out_channels*2),
                nn.Softplus(),
                nn.Linear(out_channels*2, num_classes),
            )

    def forward(self, x: Tensor, gt_voxel_semantics: Tensor = None) -> Any:
        x = rearrange(x, 'b (C Dim16) DimH DimW -> b C Dim16 DimH DimW', Dim16=16)
        occ_pred = self.final_conv(x).permute(0, 4, 3, 2, 1)
        # bncdhw->bnwhdc
        occ_pred = self.predicter(occ_pred)
        return occ_pred
    
    def get_loss(self, voxel_semantics, mask_lidar, preds):
        loss_ = dict()
        voxel_semantics=voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_lidar
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()
            loss_occ=self.loss_occ(preds,voxel_semantics,mask_camera, avg_factor=num_total_samples)
            loss_['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics,)
            loss_['loss_occ'] = loss_occ
        return loss_


class OccupancyBEVFusionCenterHead(BEVFusion):
    def __init__(self, model_cfg) -> Any:
        super().__init__(model_cfg)

    def forward(
        self,
        lidar_points,
        cameras_imgs,
        metas,
        gt_boxes,
        return_feature=False,
        **kwargs
    ):
        if self.with_lidar_encoder:
            lidar_output = self.lidar_encoder(lidar_points)
            model_output = lidar_output
        if self.with_camera_encoder:
            camera_output = self.camera_encoder(cameras_imgs, metas)
            model_output = camera_output
        if self.with_fusion_encoder:
            multimodal_output = self.fusion_encoder(lidar_output, camera_output)
            model_output = multimodal_output
        x = self.bev_encoder(model_output)

        occ_pred = self.det_head(x[0], gt_boxes)
        if return_feature == True:
            return model_output, x[0], forward_ret_dict["multi_head_features"]
        if self.training:
            voxel_semantics = kwargs['voxel_semantics']
            mask_lidar = kwargs['mask_lidar']
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17

            loss_dict = self.det_head.get_loss(voxel_semantics, mask_lidar, occ_pred)
            return (
                loss_dict['loss_occ'],
                loss_dict)
        else:
            return occ_pred

    def _configure_det_head(self):
        return OccupancyHead(self.cfg.det_head)


def _load_data_to_gpu(data_dict):
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            data_dict[k] = v.cuda()
        elif isinstance(v, dict):
            _load_data_to_gpu(data_dict[k])
        else:
            data_dict[k] = v


class Exp(BaseExp):
    def __init__(
        self,
        batch_size_per_device=4,
        total_devices=1,
        max_epochs=20,
        ckpt_path=None,
        data_cfg=DATA_CFG,
        model_cfg=MODEL_CFG,
        **kwargs
    ):
        super(Exp, self).__init__(
            batch_size_per_device, total_devices, max_epochs, ckpt_path, 
            data_cfg=data_cfg, model_cfg=model_cfg)
        
        if self.model_cfg["camera_encoder"] is not None:
            self.model_cfg["camera_encoder"]["img_backbone_conf"] = _IMG_BACKBONE_CONF
            self.model_cfg["camera_encoder"]["img_neck_conf"] = _IMG_NECK_CONF
            self.model_cfg["camera_encoder"]["depth_net_conf"] = _DEPTH_NET_CONF
        
        self.model_cfg["det_head"] = CENTERPOINT_DET_HEAD_CFG
        self._change_cfg_params()

        self.model = self._configure_model()
        
        ## build the dataloader
        print("Start build the dataloader...")
        self.train_dataloader = self.configure_train_dataloader()
        self.val_dataloader = self.configure_val_dataloader()
        # self.test_dataloader = self.configure_test_dataloader()

    def _configure_model(self):
        model = OccupancyBEVFusionCenterHead(
            model_cfg=mmcv.Config(self.model_cfg))
        return model
    
    def training_step(self, batch):
        if torch.cuda.is_available():
            _load_data_to_gpu(batch)
        if "points" in batch:
            points = [frame_point for frame_point in batch["points"]]
        else:
            points = None
        imgs = batch.get("imgs", None)
        metas = batch.get("mats_dict", None)
        voxel_semantics = batch['voxel_semantics']
        mask_lidar = batch['mask_lidar']

        batch_dict = dict(points=points, imgs=imgs, metas=metas,
                          gt_boxes=None, voxel_semantics=voxel_semantics,
                          mask_lidar=mask_lidar)

        loss, loss_dict = self(**batch_dict)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if torch.cuda.is_available():
            _load_data_to_gpu(batch)
        if "points" in batch:
            points = [frame_point for frame_point in batch["points"]]
        else:
            points = None
        imgs = batch.get("imgs", None)
        metas = batch.get("mats_dict", None)
        occ_pred = self(points, imgs, metas, None)
        occ_score=occ_pred.softmax(-1)
        occ_res=occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return occ_res
    
    def validation_epoch_end(self, outputs) -> None:
        print("validation_epoch_end")
        torch_dist.synchronize()
        if torch_dist.get_rank() == 0:
            print(self.val_dataloader.dataset.evaluate(outputs))
    
    def configure_train_dataloader(self):
        train_dataset = NuScenesDatasetOccpancy(
            **self.data_cfg['train'],
            data_split=self.data_split["train"],
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=32,
            drop_last=False,
            shuffle=False,
            collate_fn=partial(collate_fn, is_return_depth=False),
            sampler=None,
            pin_memory=False,
        )
        return train_loader

    def configure_val_dataloader(self):
        val_dataset = NuScenesDatasetOccpancy(
            **self.data_cfg['val'],
            data_split=self.data_split["val"],
        )
        print(f"The validation dataset length is {len(val_dataset)}")

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=32,
            sampler=None,
            pin_memory=False,
        )
        return val_loader

    def configure_test_dataloader(self):
        test_dataset = NuScenesDatasetOccpancy(
            **self.data_cfg['test'],
            data_split=self.data_split["test"],
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=32,
            sampler=None,
            pin_memory=False,
        )
        return test_loader


if __name__ == "__main__":
    import logging

    logging.getLogger("mmcv").disabled = True
    logging.getLogger("mmseg").disabled = True
    run_cli(Exp, "OccupancyBEVFusion_nuscenes_centerhead_fusion_exp")
