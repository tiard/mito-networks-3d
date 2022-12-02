import sys
sys.path.insert(0, 'src')
from parameters import *
import numpy as np
from tqdm import tqdm
import kimimaro
import networkx as nx
from utils import save_dict


def compute_length(im, radius=1200, outpath=None, save=False):
    if save and outpath is None:
        raise ValueError('outpath should be provided if save is passed')
       
    skels = kimimaro.skeletonize(
        im, 
        progress=True, 
        dust_threshold=0) 
    
    lengths = np.zeros((len(skels.items())))
    i = 0
    for _, skeleton in tqdm(skels.items()):
        points = skeleton.vertices
        points *= VOLUME_SCALE
        num_points = len(points)
        idx = np.arange(start=0, stop=num_points, step=1, dtype=np.int)
        pos = {k : v for (k, v) in zip(idx, points)}
        g = nx.random_geometric_graph(len(points), radius=radius, pos=pos)
        try:
            lengths[i] = nx.diameter(g)
        except nx.exception.NetworkXError:
            lengths[i] = 0
        i += 1

    lengths *= radius

    if save:
        np.save(outpath, lengths)
        outpath.replace('.npy', '.csv')
        np.savetxt(outpath, lengths, delimiter=',', fmt='%s')
        outpath.replace('.csv', '.pkl')
        save_dict(skels, outpath)
    else:
        return lengths
    
def filter_length(lengths, outpath=None, save=False):

    if save and outpath is None:
        raise ValueError('outpath should be provided if save is passed')

    lengths_filtered_lower = (lengths > LENGTH_LOWER_BOUND).astype(np.uint8)
    lengths_filtered_upper = (lengths < LENGTH_UPPER_BOUND).astype(np.uint8)

    print(
        f'Mean length: {np.mean(lengths)}, \
        Largest length: {np.max(lengths)}, \
        Smallest length: {np.min(lengths)}'
    )
    print(
        f'{np.sum(lengths_filtered_lower)} / {len(lengths)} \
        larger than lower bound'
        )
    print(
        f'{np.sum(lengths_filtered_upper)} / {len(lengths)} \
        smaller than upper bound'
    )

    selected_indices = lengths_filtered_lower * lengths_filtered_upper
    selected_lengths = lengths[selected_indices > 0]

    if save:
        np.save(outpath, lengths)
        outpath.replace('.npy', '.csv')
        np.savetxt(outpath, selected_lengths, delimiter=',', fmt='%s')

    else:
        return 
