import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pyshearlab


def shear(image_data):
	shearletSystem_norm = pyshearlab.SLgetShearletSystem2D(1,image_data.shape[0],image_data.shape[1],3)
	coeffs = pyshearlab.SLsheardec2D(image_data,shearletSystem_norm)
	result = np.sum(np.abs(coeffs), axis=2)
	return result

def largest_slice(image):
	non_empty_slices = np.any(image, axis=(0, 2))
	indices = np.where(non_empty_slices)[0]
	
	start_index = indices[0]
	end_index = indices[-1]
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

def segment(ori, aseg, a1, a2):
	left = np.where(aseg == a1, 1, 0)
	right = np.where(aseg == a2, 1, 0)
	mask = left + right
	segment = mask * ori
	return segment, mask

def process_hippo(ori, aseg):
	region, _ = segment(ori, aseg, 17, 53)
	amygdala, _ = segment(ori, aseg, 18, 54)

	non_empty_slices = np.any(amygdala, axis=(0, 1))
	indices = np.where(non_empty_slices)[0]
	i = indices[0] - 1
	ori_slice = ori[:,:,i]
	hippo_slice = region[:,:,i]
	return ori_slice, hippo_slice


def process_ventricle(ori, aseg):
	region, mask = segment(ori, aseg, 4, 43)
	i = largest_slice(mask)
	ven_slice = region[:, i, :]
	ori_slice = ori[:, i, :]
	return ori_slice, ven_slice


def save_as_jpg(output_path, image):
	plt.figure(figsize=(6, 6))
	plt.imshow(image, cmap='gray')
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
	plt.close()
	

def process_subjects_in_folder(folder_name, region):
	subjects_dir = os.listdir(folder_name)
	base_folder_name = os.path.basename(folder_name)

	
	folder_list = [f'{region}_ori', f'{region}_segment', f'{region}_shear', f'{region}_shearseg']
	for folder in folder_list:
		if not os.path.exists(f'data/{folder}/{base_folder_name}'):
			os.makedirs(f'data/{folder}/{base_folder_name}')
		if not os.path.exists(f'img/{folder}/{base_folder_name}'):
			os.makedirs(f'img/{folder}/{base_folder_name}')


	for subject in subjects_dir:
		if subject == 'fsaverage':
			continue
		mri_path = os.path.join(folder_name, subject, 'mri')
		mri_file = os.path.join(mri_path, 'brain.finalsurfs.mgz')
		aseg_file = os.path.join(mri_path, 'aseg.mgz')
  
		mri_data = nib.load(mri_file).get_fdata()
		aseg_data = nib.load(aseg_file).get_fdata()
		
		if region == "hippo":
			ori, segment = process_hippo(mri_data, aseg_data)
		elif region == "ven":
			ori, segment = process_ventricle(mri_data, aseg_data)
			
		ori_shear = shear(ori)
		seg_shear = shear(segment)
		data_list = [ori, segment, ori_shear, seg_shear]
		for data, folder in zip(data_list, folder_list):
			# Save as .npy
			output_path = os.path.join(f'data/{folder}/{base_folder_name}', f"{subject}_{region}.npy")
			np.save(output_path, data)

			# Save as .jpg
			output_path = os.path.join(f'img/{folder}/{base_folder_name}', f"{subject}_{region}.jpg")
			save_as_jpg(output_path, data)