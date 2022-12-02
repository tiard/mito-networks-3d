import os
from glob import glob
from natsort import natsorted

def combine_txt_files(txt_dir, volume_id, start_idx, end_idx):

    # List generated text files
    if isinstance(txt_dir, list):
        txt_files = []
        for t in txt_dir:
            txt_files.extend(natsorted(glob(os.path.join(t, '*.txt'))))
        
        txt_dir = txt_dir[0]
    else:
        txt_files = natsorted(glob(os.path.join(txt_dir, '*.txt')))
    
    print(f'Found {len(txt_files)} text files to combine \n')

    output_file = os.path.join(
        txt_dir, 
        f'volume_{volume_id}_slices{start_idx}-{end_idx}.txt'
        )

    writer = open(output_file, 'w')

    print(f'Writing to {output_file} \n')

    for f in txt_files:
        reader = open(f, 'r')
        data = reader.readlines()
        writer.writelines(data)
        reader.close()
    
    writer.close()

def rotate_txt_file(volume_txt_file, output_text_file):

    reader = open(volume_txt_file, 'r')
    writer = open(output_text_file, 'w')

    lines = reader.readlines()
    for line in lines:
        vals = line.split(' ')
        assert len(vals) == 6, \
            f'Line should contain 5 elements plus newline but got {line} \n'
        vals[3] = str(2014 - float(vals[3]))
        vals[-2] = str(float(vals[-2]) - 1)
        vals = ' '.join(vals)
        writer.write(vals)

def extract_slices_from_txt(volume_txt_file, output, beginning, end):

    reader = open(volume_txt_file, 'r')
    writer = open(output, 'w')

    lines = reader.readlines()
    for line in lines:
        vals = line.split(' ')
        assert len(vals) == 6, \
            f'Line should contain 5 elements plus newline but got {line} \n'
        if float(vals[-2]) > beginning and float(vals[-2]) < end + 1:
            vals = ' '.join(vals)
            writer.write(vals)


