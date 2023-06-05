'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-04 11:37:05
Email: haimingzhang@link.cuhk.edu.cn
Description: Add the occupancy related information into the original nuscenes data .pkl files.
'''

import pickle
from tqdm import tqdm
from nuscenes import NuScenes


def add_occ_info(extra_tag):
    nuscenes_version = 'v1.0-trainval'
    dataroot = './data/nuscenes/'
    nuscenes = NuScenes(nuscenes_version, dataroot)

    for set in ['train', 'val']:
        with open("./data/nuscenes/{}_info.pkl".format(set), "rb") as f:
            infos = pickle.load(f)

        for id in tqdm(range(len(infos))):
            if id % 10 == 0:
                print('%d/%d' % (id, len(infos)))
            info = infos[id]
            scene = nuscenes.get('scene', info['scene_token'])
            infos[id]['occ_path'] = \
                './data/nuscenes/gts/%s/%s'%(scene['name'], info['sample_token'])
        with open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set),
                  'wb') as fid:
            pickle.dump(infos, fid)
    

if __name__ == "__main__":
    extra_tag = 'occ-nuscenes'
    add_occ_info(extra_tag)
