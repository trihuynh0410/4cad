import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pyshearlab
import sys
from utils import *
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    region = sys.argv[1] # region = hippo or ven
    for folder in  ["CN", "MCI", "Mild", 'Mod']:
        process_subjects_in_folder(f'data/ADNI_out_mgz/{folder}', region) 
        print(f"Done {folder}")