_POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]
_VOXEL_SIZE = [0.05, 0.05, 0.16]
_GRID_SIZE = [1600, 1600, 40]
_IMG_DIM = (256, 704)
_OUT_SIZE_FACTOR = 8

COMMON_CFG = dict(
    point_cloud_range=_POINT_CLOUD_RANGE,
    voxel_size=_VOXEL_SIZE,
    grid_size=_GRID_SIZE,
    img_dim=_IMG_DIM,
    out_size_factor=_OUT_SIZE_FACTOR,
)

CLASS_NAMES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

_AUG_CFG = dict(
    point_cloud_range=_POINT_CLOUD_RANGE,
    img_norm_cfg=dict(
        img_mean=[123.675, 116.28, 103.53], img_std=[58.395, 57.12, 57.375], to_rgb=True
    ),
    ida_aug_cfg=dict(
        resize_lim=(0.386, 0.55),
        final_dim=_IMG_DIM,
        rot_lim=(-5.4, 5.4),
        H=900,
        W=1600,
        rand_flip=True,
        bot_pct_lim=(0.0, 0.0),
    ),
    bda_aug_cfg=dict(
        rot_lim=(-22.5 * 2, 22.5 * 2),
        scale_lim=(0.90, 1.10),
        trans_lim=(0.5, 0.5, 0.5),
        flip_dx_ratio=0.5,
        flip_dy_ratio=0.5,
    ),
)

val_data_config = dict(
    anno_file="data/nuscenes/occ-nuscenes_infos_val.pkl",
    root_path="./data/nuscenes",
    lidar_key_list=["LIDAR_TOP"],
    img_key_list=[
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
    ],
    num_lidar_sweeps=10,
    num_cam_sweeps=0,
    lidar_with_timestamp=True,
    class_names=CLASS_NAMES,
    use_cbgs=False,
    aug_cfg=_AUG_CFG)

DATA_CFG = dict(
    train=dict(
        anno_file="data/nuscenes/occ-nuscenes_infos_train.pkl",
        root_path="./data/nuscenes",
        lidar_key_list=["LIDAR_TOP"],
        img_key_list=[
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
        ],
        num_lidar_sweeps=10,
        num_cam_sweeps=0,
        lidar_with_timestamp=True,
        class_names=CLASS_NAMES,
        use_cbgs=False,
        aug_cfg=_AUG_CFG),
    val=val_data_config,
    test=val_data_config
)

MODEL_CFG = dict(
    class_names=CLASS_NAMES,
    lidar_encoder=dict(
        point_cloud_range=_POINT_CLOUD_RANGE,
        voxel_size=_VOXEL_SIZE,
        grid_size=_GRID_SIZE,
        max_num_points=10,
        max_voxels=(120000, 160000),
        src_num_point_features=5,
        use_num_point_features=5,
        map_to_bev_num_features=256,
    ),
    camera_encoder=dict(
        x_bound=[
            _POINT_CLOUD_RANGE[0],
            _POINT_CLOUD_RANGE[3],
            _VOXEL_SIZE[0] * _OUT_SIZE_FACTOR,
        ],
        y_bound=[
            _POINT_CLOUD_RANGE[1],
            _POINT_CLOUD_RANGE[4],
            _VOXEL_SIZE[1] * _OUT_SIZE_FACTOR,
        ],
        z_bound=[
            _POINT_CLOUD_RANGE[2],
            _POINT_CLOUD_RANGE[5],
            _POINT_CLOUD_RANGE[5] - _POINT_CLOUD_RANGE[2],
        ],
        d_bound=[2.0, 58.0, 0.5],
        final_dim=_IMG_DIM,
        output_channels=256,
        downsample_factor=16,
        img_backbone_conf=dict(
            type="SwinTransformer",
            embed_dims=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=[1, 2, 3],
            with_cp=False,
            convert_weights=True,
            init_cfg=dict(
                type="Pretrained",
                checkpoint="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
            ),
        ),
        img_neck_conf=dict(
            type="SECONDFPN",
            in_channels=[192, 384, 768],
            upsample_strides=[0.5, 1, 2],
            out_channels=[128, 128, 128],
        ),
        depth_net_conf=dict(in_channels=384, mid_channels=384),
    ),
    bev_encoder=dict(
        backbone2d_layer_nums=[5, 5],
        backbone2d_layer_strides=[1, 2],
        backbone2d_num_filters=[128, 256],
        backbone2d_upsample_strides=[1, 2],
        backbone2d_num_upsample_filters=[256, 256],
        num_bev_features=256,  # sp conv output channel
        backbone2d_use_scconv=False,
    ),
    det_head=dict(
        target_assigner=dict(
            point_cloud_range=_POINT_CLOUD_RANGE,
            voxel_size=_VOXEL_SIZE,
            grid_size=_GRID_SIZE,
            gaussian_overlap=0.1,
            min_radius=2,
            iou_calculator=dict(type="BboxOverlaps3D", coordinate="lidar"),
            cls_cost=dict(type="FocalLossCost", gamma=2, alpha=0.25, weight=0.15),
            reg_cost=dict(type="BBoxBEVL1Cost", weight=0.25),
            iou_cost=dict(type="IoU3DCost", weight=0.25),
        ),
        bbox_coder=dict(
            pc_range=_POINT_CLOUD_RANGE[0:2],
            voxel_size=_VOXEL_SIZE[0:2],
            out_size_factor=_OUT_SIZE_FACTOR,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10,
        ),
        dataset_name="nuScenes",
        num_proposals=200,
        hidden_channel=128,
        in_channels=512,
        num_classes=len(CLASS_NAMES),
        num_decoder_layers=1,
        num_heads=8,
        nms_kernel_size=3,
        out_size_factor=_OUT_SIZE_FACTOR,
        common_heads=dict(
            center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
        ),
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
    ),
)

CENTERPOINT_DET_HEAD_CFG = dict(
    in_channels=32,
    out_channels=32,
    num_classes=18
)
