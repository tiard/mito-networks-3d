import os
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import numpy as np
from scipy.spatial.distance import cdist
import os
import numpy as np
from scipy import spatial as sp
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image
from skimage.measure import regionprops
import pickle

def draw_image(name, array):
    plt.cla()
    plt.clf()
    plt.figure()
    plt.title(os.path.basename(name))
    plt.axis('off')
    plt.imshow(array)
    plt.savefig(name)
    plt.close()

def recolor_array_single_threaded(image, ref, past_idx):

    ref_regions = regionprops(ref)
    im_regions = regionprops(image)

    ref_cent = [p['centroid'] for p in ref_regions]
    im_cent = [p['centroid'] for p in im_regions]

    ref_lbl = [p['label'] for p in ref_regions]

    new_image = np.zeros_like(image)

    dist = cdist(im_cent, ref_cent)
    matches = np.argmin(dist, axis=1)

    for i, p in enumerate(im_regions):

        # Get reference object
        ref_object = ref_regions[matches[i]]

        # Check that overlap with ref object is nonzero
        im_mask = np.zeros_like(new_image)
        min_row, min_col, max_row, max_col = p['bbox']
        im_mask[min_row:max_row, min_col:max_col] = p['image']

        ref_mask = np.zeros_like(new_image)
        min_row, min_col, max_row, max_col = ref_object['bbox']
        ref_mask[min_row:max_row, min_col:max_col] = ref_object['image']

        if np.count_nonzero(ref_mask * im_mask) > 0:
            new_image[p['coords'][:, 0], p['coords'][:, 1]] = \
                ref_lbl[matches[i]]
        
        # Mark new object if no overlap
        else:
            new_index = max(np.max(past_idx), np.max(ref_lbl)) + 1
            new_image[p['coords'][:, 0], p['coords'][:, 1]] = new_index
    
    new_values = set(np.unique(new_image)[1:])
    ref_set = set(ref_lbl)
    intersection = ref_set.intersection(new_values)
    retired = ref_set.difference(intersection)

    past_idx.extend(list(retired))

    return new_image, past_idx

def relabel_object(idx, ref_object, p, ref_lbl, image, past_idx, ret):

    # Array to return
    new_image = np.zeros_like(image)

    # Check that overlap with ref object is nonzero
    im_mask = np.zeros_like(new_image)
    min_row, min_col, max_row, max_col = p['bbox']
    im_mask[min_row:max_row, min_col:max_col] = p['image']

    ref_mask = np.zeros_like(new_image)
    min_row, min_col, max_row, max_col = ref_object['bbox']
    ref_mask[min_row:max_row, min_col:max_col] = ref_object['image']

    if np.count_nonzero(ref_mask * im_mask) > 0:
        new_image[p['coords'][:, 0], p['coords'][:, 1]] = ref_lbl
    
    # Mark new object if no overlap
    else:
        new_index = max(np.max(past_idx), np.max(ref_lbl)) + 1
        new_image[p['coords'][:, 0], p['coords'][:, 1]] = new_index
    
    ret[idx] = new_image

def randomize_to_rgb(image):
    vals = np.unique(image)[1:]
    h, w = image.shape
    ret = np.zeros([h, w, 3], dtype=np.uint8)
    
    for v in vals:
        color = np.random.randint(
            low=0,
            high=255,
            size=3,
            dtype=np.uint8
        )
        ret[image==v] += color
    
    return ret

def nearest_neighbor(mito, nuclei):
    ret = np.zeros_like(nuclei)
    
    mvals = np.unique(mito)[1:]
    cvals = np.unique(nuclei)[1:]
    
    nuclei_pixels = []
    for c in cvals:
        nx, ny = np.where(nuclei == c)
        nu_pix = np.stack([nx, ny], axis=1)
        nuclei_pixels.append(nu_pix)
    
    distances_mito_nuclei = {}
    
    for c in cvals:
        distances_mito_nuclei[c] = []
    
    for v in mvals:
        mx, my = np.where(mito == v)
        mp = np.stack([mx, my], axis =1)
        dist = np.inf
        
        for i, c in enumerate(nuclei_pixels):
            d = np.mean(cdist(mp, c, metric='euclidean'))
            if d < dist:
                dist = d
                idx = i
        
        current_distances = distances_mito_nuclei[cvals[idx]]
        current_distances.append(dist)
        distances_mito_nuclei[cvals[idx]] = current_distances
        
        ret[mito == v] = cvals[idx]
        
    return ret, distances_mito_nuclei

def nearest_neighbor_centroid(mito, nuclei):
    
    mprops = regionprops(mito)
    cprops = regionprops(nuclei)
    
    m_centroid = np.asarray([p['centroid'] for p in mprops])
    c_centroid = np.asarray([p['centroid'] for p in cprops])
    
    d = sp.distance.cdist(m_centroid, c_centroid, metric='euclidean')
    
    mito_assignments = np.argmin(d, axis=1)
    
    # First nuclei starts at 1, 0 is background
    mito_assignments = np.insert(mito_assignments, 0, 0)
    mito_assignments += 1
    
    print(np.min(mito_assignments), np.max(mito_assignments))
    print(np.min(mito), np.max(mito))
    print(np.min(nuclei), np.max(nuclei))
    mito_assigned_map = mito_assignments[mito] * (mito !=0).astype(np.uint8)
    
    return mito_assigned_map, mito_assignments

def object_convex_hull(image):
    vals = np.unique(image)[1:]
    ret_arr = np.zeros_like(image)
    
    for v in vals:
        cur_obj = (image == v).astype(np.int)
        chull = convex_hull_image(cur_obj)
        ret_arr[chull != 0] = v
    
    return ret_arr

def separate_mito_nuclei(label_image):
    # Split nuclei and mitochondria channels
    lbl_vals = np.unique(label_image)

    mito = (label_image == lbl_vals[2]).astype(np.uint8)

    nuclei = (label_image == lbl_vals[1]).astype(np.uint8)

    return nuclei, mito

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_val = pickle.load(f)
    return ret_val
