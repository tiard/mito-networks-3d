import os, sys
sys.path.insert(0, 'mito_length')
sys.path.insert(0, 'total_distance')
sys.path.insert(0, 'mito_volume')
sys.path.insert(0, 'mito_sphericity')
import numpy as np
from glob import glob
import argparse
from utils import separate_mito_nuclei

parser = argparse.ArgumentParser()

parser.add_argument('--input_volume', type=str)
parser.add_argument('--output_directory', type=str)
parser.add_argument('--tasks', nargs='+', type=str)
parser.add_argument('--filter', action='store_true')

args = parser.parse_args()

# Generate output structure
if not os.path.exists(args.output_directory):
    os.mkdir(args.output_directory)

# Load volume
vol = np.load(args.input_volume)
mito_arr, nuclei_arr = separate_mito_nuclei(vol) 

# Compute parameters and save
if 'volume' in args.tasks:
    volume_outpath = os.path.join(args.output_directory, 'volumes.npy')
    mito_volume.compute_volume(mito_arr, volume_outpath, save=True)
if 'length' in args.tasks:
    length_outpath = os.path.join(args.output_directory, 'lengths.npy')
    mito_length.compute_length(mito_arr, length_outpath, save=True)
if 'sphericity' in args.tasks:
    sphericity_outpath = os.path.join(args.output_directory, 'sphericity.npy')
    mito_sphericity.compute_sphericity(mito_arr, sphericity_outpath, save=True)
if 'total_distance' in args.tasks:
    distance_outpath = os.path.join(args.output_directory, 'distances.npy')
    all_distances.compute_total_distances(
        mito_arr, 
        nuclei_arr, 
        distance_outpath, 
        save=True
    )
if 'to_imod' in args.tasks:
    volume_to_txt(
        vol, 
        args.output_directory, 
        'mito_nucleus', 
        remove_holes=False, 
        min_slice=0,
        max_slice=-1
    )