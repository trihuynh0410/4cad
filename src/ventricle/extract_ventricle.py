import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pyshearlab
from scipy.ndimage import rotate
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def shear(image_data):
	shearletSystem_norm = pyshearlab.SLgetShearletSystem2D(1,image_data.shape[0],image_data.shape[1],3)
	coeffs = pyshearlab.SLsheardec2D(image_data,shearletSystem_norm)
	result = np.sum(np.abs(coeffs), axis=2)
	return result

def get_largest_area_slice(image):

	non_empty_slices = np.any(image, axis=(0, 2))
	indices = np.where(non_empty_slices)[0]
	
	start_index = indices[0] + 0.2 * (indices[-1] - indices[0])
	end_index = indices[0] + 0.39 * (indices[-1] - indices[0])
	
	start_index = int(np.floor(start_index))
	end_index = int(np.ceil(end_index))
	
	largest_area = 0
	largest_area_slice = start_index
	for i in range(start_index, end_index + 1):
		area = np.sum(image[:, i, :])
		if area > largest_area:
			largest_area = area
			largest_area_slice = i
			
	return largest_area_slice

def get_middle_slice_index(image):
	non_empty_slices = np.any(image, axis=(0, 2))
	first_slice = np.where(non_empty_slices)[0][0]
	last_slice = np.where(non_empty_slices)[0][-1]
	middle_slice = first_slice + 60
	middle_slice = int(np.ceil(middle_slice))
	return middle_slice

def segment_and_extract_middle_slice(mri_array, aseg_array):
	left_ventricle_mask = np.where(aseg_array == 4, 1, 0)
	right_ventricle_mask = np.where(aseg_array == 43, 1, 0)
	ventricles_mask = left_ventricle_mask + right_ventricle_mask
	extracted_ventricles = ventricles_mask * mri_array
	middle_slice_index = get_largest_area_slice(ventricles_mask)
	#middle_slice_index = get_middle_slice_index(extracted_ventricles)
	extracted_middle_slice = extracted_ventricles[:, middle_slice_index, :]
	return extracted_middle_slice

def save_as_jpg(image, output_path):
	plt.figure(figsize=(6, 6))
	plt.imshow(image, cmap='gray')
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
	plt.close()

def process_subjects_in_folder(folder_name):
	subjects_dir = os.listdir(folder_name)
	base_folder_name = os.path.basename(folder_name)

	# Create separate directories for each class
	if not os.path.exists(f'ventricles_shearseg_jpgs/{base_folder_name}'):
		os.makedirs(f'ventricles_shearseg_jpgs/{base_folder_name}')

	if not os.path.exists(f'ventricles_testshearseg/{base_folder_name}'):
		os.makedirs(f'ventricles_shearseg/{base_folder_name}')

	for subject in subjects_dir:
		if subject == 'fsaverage':
			continue
		mri_path = os.path.join(folder_name, subject, 'mri')
		mri_file = os.path.join(mri_path, 'brain.finalsurfs.mgz')
		aseg_file = os.path.join(mri_path, 'aseg.mgz')

		mri_data = nib.load(mri_file).get_fdata()
		aseg_data = nib.load(aseg_file).get_fdata()

		extracted_slice = segment_and_extract_middle_slice(mri_data, aseg_data)
		extracted_slice = shear(extracted_slice)
		# Save as jpg
		output_path_jpg = os.path.join(f'ventricles_shearseg_jpgs/{base_folder_name}', f"{subject}_shearseg_ventricle.jpg")
		save_as_jpg(extracted_slice, output_path_jpg)

		# Save as numpy array
		output_path_npy = os.path.join(f'ventricles_shearseg/{base_folder_name}', f"{subject}_shearseg_ventricle.npy")
		np.save(output_path_npy, extracted_slice)

if __name__ == "__main__":

	for folder in ['CN', 'MCI', 'Mild', 'Mod']:
		process_subjects_in_folder(f'ADAI_out/{folder}')

