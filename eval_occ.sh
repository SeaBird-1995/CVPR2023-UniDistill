# CHECKPOINT=checkpoints/lidar2camera/checkpoint/l2c_submit.pth
# CHECKPOINT=checkpoints/camera2lidar/checkpoint/c2l_submit.pth

set -x
python unidistill/exps/multisensor_fusion/nuscenes/Occupancy/occupancy_nuscenes_centerhead_lidar_exp.py -b 1 --gpus 1 -e