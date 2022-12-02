import os
from skimage.io import imread
import numpy as np
from skimage.color import rgba2rgb, rgb2gray
from scipy import ndimage as nd
from natsort import natsorted
from glob import glob
import cv2
from skimage.measure import find_contours
from tqdm import tqdm
import utils

def slices_to_txt(
    datadir, 
    output_dir, 
    replace_with_gt=True,
    min_slice=0,
    max_slice=-1):

    # Define image and label paths
    imagedir = os.path.join(datadir, 'image')
    labeldir = os.path.join(datadir, 'output_label')

    if replace_with_gt:
        gtdir = os.path.join(datadir, 'ground_truth_label')

    # Set up output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # List files
    images = natsorted(glob(imagedir + '/*'))
    labels = natsorted(glob(labeldir + '/*'))

    if replace_with_gt:
        gt = natsorted(glob(gtdir + '/*'))

    # Filter images for which we have labels and ground truth
    imnames = [os.path.basename(f) for f in images]
    lnames = [os.path.basename(f) for f in labels]
    if replace_with_gt:
        gtnames = [os.path.basename(f) for f in gt]

    if replace_with_gt:
        # Optionally replace predictions with ground truth 
        gt_found = 0
        for gn in gtnames:
            idx = lnames.index(gn)
            lnames[idx] = gn
            gt_found += 1 

        print(f'Found {gt_found} ground truths')

    imnames = natsorted([f for f in imnames if f in lnames])

    # Add path back
    images = [os.path.join(imagedir, f) for f in imnames]

    print(f'Found {len(images)} images and {len(labels)} labels\n')

    if max_slice == -1:
        max_slice = len(images)

    # Generate output names
    output_paths = [os.path.join(
        output_dir, f.replace('.png', '.txt').replace('.tif', '.txt')
        ) for f in imnames]

    # Reassign max slice
    if max_slice == -1:
        max_slice = len(images)

    # Define class id
    mito_nuc_class_ids = [1, 2]
    cell_class_ids = [1]

    # Iterate through slices and write text file
    for k in tqdm(range(min_slice, max_slice + 1)):

        # We want to 1 index everything
        slice_idx = k + 1 

        # Load label
        im = (imread(images[k]) * 255)
        lbl = imread(labels[k])

        if lbl.shape[-1] == 4:
            lbl = rgba2rgb(lbl)

        if lbl.shape[-1] == 3:
            lbl = (rgb2gray(lbl) * 255).astype(np.uint8)

        # Check that the dimensions are the same
        imshape = im.shape
        h = imshape[0]
        w = imshape[1]

        hl, wl = lbl.shape

        if hl != h  or wl != w:
            lbl = cv2.resize(
                lbl,
                dsize=(w, h),
                interpolation=cv2.INTER_NEAREST)

        # Create output file
        writer = open(output_paths[k], 'w')

        # Separate object semantics
        if len(np.unique(lbl)) > 2:
            cell = False
            nuclei, mito = utils.separate_mito_nuclei(label_image=lbl)
            class_ids = mito_nuc_class_ids
        else:
            cell = True
            class_ids = cell_class_ids

        # Label instances
        if cell:
            cell_instances = nd.label(lbl)[0]
            print(f'Found {np.max(cell_instances)} cells in slice {k}')
            objects = [cell_instances]
        else:
            nuclei_instances = nd.label(nuclei)[0]
            mito_instances = nd.label(mito)[0]
            print(f'Found {np.max(nuclei_instances)} nuclei and \
                {np.max(mito_instances)} mitochondria in slice {k}')
            objects = [nuclei_instances, mito_instances]
        
        for o_id, o in enumerate(objects):
            vals = np.unique(o)[1:]

            # Iterate over each object and write to text
            for i, v in tqdm(enumerate(vals), total=np.max(vals)):
        
                # Binary image
                target = np.where(o == v, 1, 0)

                # Current class id
                curr_id = np.max(class_ids) * slice_idx - \
                    (np.max(class_ids) - class_ids[o_id])

                # Get contours for the current object
                contours = find_contours(target, 0.999)

                # if no countours found, continue
                if len(contours) == 0:
                    continue
            
                if len(contours) > 1:
                    max_len = -1
                    for i, c in enumerate(contours):
                        if len(c) > max_len:
                            max_idx = i
                            max_len = len(c)
                    contour = contours[max_idx]
                else:
                    contour = contours[0]

                for p in contour:
                    writer.write(
                        f'{curr_id} {v} {p[1]} {h - p[0]} {slice_idx - 0.5} \n'
                    )
            
        writer.close()