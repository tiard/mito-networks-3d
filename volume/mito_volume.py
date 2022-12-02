import sys
sys.path.insert(0, 'src')
from parameters import *
import numpy as np
from tqdm import tqdm
import porespy


def compute_volume(im, outpath=None, save=False):

    if save and outpath is None:
        raise ValueError('outpath should be provided if save is passed')

    regions = porespy.metrics.regionprops_3D(im)
    volumes = np.zeros(len(regions))
    for i, r in tqdm(enumerate(regions), total=np.max(im)):
        volumes[i] = r.volume * VOXEL_VOLUME

    if save:
        np.save(outpath, volumes)
        outpath.replace('.npy', '.csv')
        np.savetxt(outpath, volumes, delimiter=',', fmt='%s')
    else:
        return volumes


def filter_volume(volumes, outpath=None, save=False):

    if save and outpath is None:
        raise ValueError('outpath should be provided if save is passed')

    volumes_filtered_lower = (volumes > VOLUME_LOWER_BOUND).astype(np.uint8)
    volumes_filtered_upper = (volumes < VOLUME_UPPER_BOUND).astype(np.int8)

    print(
        f'Mean Volume: {np.mean(volumes)}, \
        Largest Volume: {np.max(volumes)}, \
        Smallest Volume: {np.min(volumes)}'
    )
    print(
        f'{np.sum(volumes_filtered_lower)} / {len(volumes)} \
        larger than lower bound'
        )
    print(
        f'{np.sum(volumes_filtered_upper)} / {len(volumes)} \
        smaller than upper bound'
    )

    selected_indices = volumes_filtered_lower * volumes_filtered_upper
    selected_volumes = volumes[selected_indices > 0]

    if save:
        np.save(outpath, volumes)
        outpath.replace('.npy', '.csv')
        np.savetxt(outpath, selected_volumes, delimiter=',', fmt='%s')

    else:
        return 
