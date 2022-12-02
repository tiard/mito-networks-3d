import os
import numpy as np
from skimage.morphology import remove_small_holes
from skimage.measure import find_contours
from tqdm import tqdm

def volume_to_txt(
    volume_file, 
    output_dir, 
    datatype, 
    remove_holes=False, 
    min_slice=0,
    max_slice=-1):
    
    # Create output directories
    txt_dir = os.path.join(
        output_dir, 
        f'txt_{datatype}_{min_slice}_{max_slice}'
        )

    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)
    
    # Load volume
    if isinstance(volume_file, str):
        volume = np.load(volume_file)
    else:
        volume = volume_file

    c, h, _ = volume.shape

    if max_slice == -1:
        max_slice = c

    # def volume_to_txt(volume, txt_dir):
    vals = np.unique(volume)[1:]

    # Define class id
    if datatype == 'mito':
        class_id = 2
        class_ids = [1, 2]
    elif datatype == 'nucleus':
        class_id = 1
        class_ids = [1, 2]
    elif datatype == 'cell':
        class_id = 1
        class_ids = [1]

    # Iterate over each object and write to text
    for i, v in tqdm(enumerate(vals), total=np.max(vals)):
        
        # Create txt file
        txt_name = os.path.join(txt_dir, f'{i}.txt')
        txt_file = open(txt_name, 'w+')

        # Binary cell image
        target = np.where(volume == v, 1, 0)

        if remove_holes:
            target = remove_small_holes(target).astype(np.uint8)
        
        # Find range of valid slices
        target_pixels_per_slice = np.sum(target, axis=(1,2))
        # print(target_pixels_per_slice)
        target_slice_range = np.where(target_pixels_per_slice != 0)
        start = np.min(target_slice_range)
        end = np.max(target_slice_range)

        # assert target_slice_range == np.arange(start, end), \
            # f'Object range is not contiguous: {target_slice_range}'

        # Iterate through slices and write contours
        bounded_start = max(min_slice - 1, start)
        bounded_end = min(max_slice - 1, end)

        for slice in range(bounded_start, bounded_end):
            
            # We want to 1 index everything
            slice_idx = slice + 1 

            # Current class id
            curr_id = np.max(class_ids) * slice_idx - \
                (np.max(class_ids) - class_id)

            # print(f'Object value bounds: {np.max(target)}, {np.min(target)}')

            contours = find_contours(target[slice], 0.999)

            # if no countours found, continue
            if len(contours) == 0:
                continue
            # assert len(contours) != 0, \
                # f'Did not find a contour for {datatype} {i} at slice {slice}'
            
            # TODO : add this as an option, instance vs semantics
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
                txt_file.write(
                    f'{curr_id} {v} {p[1]} {h - p[0]} {slice_idx - 0.5} \n'
                    )
        
        txt_file.close()
    
    return txt_dir