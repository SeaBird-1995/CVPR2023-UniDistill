CHECKPOINT=checkpoints/lidar2camera/checkpoint/l2c_submit.pth
CHECKPOINT=checkpoints/camera2lidar/checkpoint/c2l_submit.pth
# python unidistill/exps/multisensor_fusion/nuscenes/BEVFusion/BEVFusion_nuscenes_centerhead_camera_exp.py -b 1 --gpus  1 -e  --ckpt_path $CHECKPOINT

set -x
python unidistill/exps/multisensor_fusion/nuscenes/BEVFusion/BEVFusion_nuscenes_centerhead_lidar_exp.py -b 1 --gpus 1 -e  --ckpt_path $CHECKPOINT