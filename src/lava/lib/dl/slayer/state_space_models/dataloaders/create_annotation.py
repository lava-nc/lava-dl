import os
import sys
import numpy as np
from tqdm import tqdm

base_annotation_file = sys.argv[1]
target_annotation_file = sys.argv[2]

with open(base_annotation_file, 'r') as f:
    base_annotations = f.readlines()


for row in tqdm(base_annotations):
    fn, start_frame, end_frame, label = row.split(" ")

    data_folder = "data_eff_features"
    if int(label) > 60:
        data_folder += "_120"

    path = f"/ssd2/users/pweidel/datasets/NTU/{data_folder}/{fn}.avi.dat.npy"
    
    data = np.load(path)
    end_frame = data.shape[0]

    with open(target_annotation_file, 'a+') as f:
        f.write(f"{path} {start_frame} {end_frame} {label}")
