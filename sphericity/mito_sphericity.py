import numpy as np
from tqdm import tqdm
import porespy


def compute_sphericity(im, outpath=None, save=False):
    
    if save and outpath is None:
        raise ValueError('outpath should be provided if save is passed')

    regions = porespy.metrics.regionprops_3D(im)
    sphericities = np.zeros(len(regions))
    for i, r in tqdm(enumerate(regions), total=np.max(im)):
        sphericities[i] = r.sphericity

    if save:
        np.save(outpath, sphericities)
        outpath.replace('.npy', '.csv')
        np.savetxt(outpath, sphericities, delimiter=',', fmt='%s')
    
    else:
        return sphericities

