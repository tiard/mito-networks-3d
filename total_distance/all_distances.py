import sys
sys.path.insert(0, 'src')
from parameters import *
from utils import save_dict
import numpy as np
from tqdm import tqdm
import porespy


def compute_total_distances(mito_im, 
                           nuclei_im, 
                           outpath=None, 
                           save=False):

    if save and outpath is None:
        raise ValueError('outpath should be provided if save is passed')

    nuclei_regions = porespy.metrics.regionprops_3D(nuclei_im)
    nuclei_centroids = np.zeros(len(nuclei_regions))
    for i, r in tqdm(enumerate(nuclei_regions), total=np.max(nuclei_im)):
        nuclei_centroids[i] = r.centroid

    mito_regions = porespy.metrics.regionprops_3D(mito_im)
    mito_centroids = np.zeros(len(mito_regions))
    for i, r in tqdm(enumerate(mito_regions), total=np.max(mito_im)):
        mito_centroids[i] = r.centroid

    mito_assignments = np.zeros((len(mito_centroids), 2))
    for i, mc in tqdm(enumerate(mito_centroids)):
        diff = np.linalg.norm((nuclei_centroids - mc) * VOLUME_SCALE, axis=1)
        diff = diff
        idx = np.argmin(diff)
        mito_assignments[i, 0] = idx
        mito_assignments[i, 1] = diff[idx]
    
    total_distances = []
    for i in range(len(nuclei_centroids)):
        correspondences = np.where(mito_assignments[:, 0] == i)
        total_distance = mito_assignments[correspondences, 1]
        total_distances.append(total_distance)

    if save:
        np.save(outpath, total_distances)
        outpath.replace('.npy', '.csv')
        np.savetxt(outpath, total_distances, delimiter=',', fmt='%s')
        outpath.replace('.csv', '.pkl')
        save_dict(total_distances, outpath)
    else:
        return total_distances