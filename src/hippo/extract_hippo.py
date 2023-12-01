import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def process_hippo(hip, dt):
	max_area = 0
	max_i = 0
	hip = hip * dt
	for i in range(0,255):
		image = hip[:, :, i]
		area = np.sum(image > 0)
		if area > max_area:
			max_area = area
			max_i = i
	max_slice = hip[:, :, max_i]
	return max_slice

def save_as_jpg(image, output_path):
	plt.figure(figsize=(6, 6))
	plt.imshow(image, cmap='gray')
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
	plt.close()

def process_subjects_in_folder(folder_name):
	subjects_dir = os.listdir(folder_name)
	
	# Create separate directories for each class
	if not os.path.exists(f'hippo_jpgs/{folder_name}'):
		os.makedirs(f'hippo_jpgs/{folder_name}')
		
	if not os.path.exists(f'hippo_npy/{folder_name}'):
		os.makedirs(f'hippo_npy/{folder_name}')
		
	for subject in subjects_dir:
		if subject == 'fsaverage':
			continue
		mri_path = os.path.join(folder_name, subject, 'mri')
		mri_file = os.path.join(mri_path, 'brain.finalsurfs.mgz')
		lh = os.path.join(mri_path, 'lh.hippoAmygLabels-T1.v21.FSvoxelSpace.mgz')
		rh = os.path.join(mri_path, 'rh.hippoAmygLabels-T1.v21.FSvoxelSpace.mgz')
		
		mri_data = nib.load(mri_file).get_fdata()
		lh_data = nib.load(lh).get_fdata()
		rh_data = nib.load(rh).get_fdata()
		hip = np.where(lh_data, lh_data, rh_data)
		hip = np.where(hip != 0, 1, 0)
		extracted_slice = process_hippo(hip, mri_data)
		
		# Save as jpg
		output_path_jpg = os.path.join(f'hippo_jpgs/{folder_name}', f"{subject}_hippo.jpg")
		save_as_jpg(extracted_slice, output_path_jpg)

		# Save as numpy array
		output_path_npy = os.path.join(f'hippo_npy/{folder_name}', f"{subject}_hippo.npy")
		np.save(output_path_npy, extracted_slice)

if __name__ == "__main__":
	
	for folder in ['CN', 'MCI', 'Mild', 'Mod']:
		process_subjects_in_folder(f'ADAI_out/{folder}')

