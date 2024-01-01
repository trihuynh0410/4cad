import os
import sys
import re
import numpy as np
import pandas as pd
import cupy as cp
from entropy import *
import time
LABELS = {
	'CN': 0,
	'MCI': 1,
	'Mild': 2,
	'Mod': 3
}


def calculate_entropy_and_profile(func, *args, **kwargs):

	result = func(*args, **kwargs)
	return result

def calculate_entropies(data, window):
    entropies = {}
    
    entropies['SampEn'], _ = calculate_entropy_and_profile(SampEn2D, data, m=window, tau=1, r=0.5, Lock=False)

    entropies['FuzzEn'] = calculate_entropy_and_profile(FuzzEn2D, data, m=window, tau=1, r=0.5, Logx=cp.exp(1), Lock=False)
    
    entropies['DispEn'], _ = calculate_entropy_and_profile(DispEn2D, data, m=window, tau=1, c=2, Typex='NCDF', Logx=cp.exp(1), Norm=False, Lock=False)
    
    entropies['DistEn'] = calculate_entropy_and_profile(DistEn2D, data, m=window, tau=1, Logx=cp.exp(1), Norm=int(0), Lock=False)
    
    entropies['PermEn'] = calculate_entropy_and_profile(PermEn2D, data, m=window, tau=1, Logx=cp.exp(1), Norm=False, Lock=False)
    
    entropies['EspEn'] = calculate_entropy_and_profile(EspEn2D, data, m=window, tau=1, Logx=cp.exp(1), Lock=False)
    
    return entropies

def process_directory(feature_dir, scales, windows):
	feature_name = os.path.basename(feature_dir)
	output_filename = f"data/entropy/Entropy_{feature_name}.csv"

	if os.path.exists(output_filename):
		results_df = pd.read_csv(output_filename)
	else:
		results_df = pd.DataFrame()

	for label, label_value in LABELS.items():
		label_dir = os.path.join(feature_dir, label)
		for subject_file in os.listdir(label_dir):
			subject_name = os.path.splitext(subject_file)[0].rsplit('_', 1)[0]

			data = np.load(os.path.join(label_dir, subject_file))
			record = {'subject': subject_name, 'label': label_value}
			for scale in scales:
				for window in windows:
					column_prefix = f"{feature_name}_Scale{scale}_Window{window}_"
					entropy_columns = [f"{column_prefix}{entropy}" for entropy in ['SampEn', 'FuzzEn', 'DispEn', 'DistEn', 'PermEn', 'EspEn']]
					
					if 'subject' in results_df.columns and subject_name in results_df['subject'].values:
						subject_row = results_df[results_df['subject'] == subject_name]
						if not subject_row[entropy_columns].isnull().values.any():
							print(f'Subject {subject_name}, Scale {scale}, Window {window} already processed. Skipping...')
							continue

					
					data_coarse = coarse_grain(data, scale)
					entropies = calculate_entropies(data_coarse, window)
					for entropy_name, entropy_value in entropies.items():
						column_name = f"{column_prefix}{entropy_name}"
						record[column_name] = entropy_value

			# results_df = results_df._append(record, ignore_index=True)
			results_df = pd.concat([results_df, pd.DataFrame([record])], ignore_index=True)
			results_df.to_csv(output_filename, index=False)
			print('Done subject:', subject_name, 'Label:', label)

	print('All done!')
if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: python 2DMSE.py <feature_directory>")
		sys.exit(1)

	feature_directory = sys.argv[1]
	scales = [3, 4]
	windows = [3, 4]
	start_time = time.time() 
	process_directory(feature_directory, scales, windows)
	end_time = time.time()  
	total_time = end_time - start_time 
	print(f"Total run time: {total_time} seconds")

