import nibabel as nib
import glob
import json
import os
import shutil

# Define paths and a flag for training
img_path = '/scratch/slurm-user3/changsun/data/rectal_MRI_volume/'
txt_path = '/scratch/slurm-user3/changsun/data/rectal_MRI_label/202301_MRI_impression_final.json'
is_train = True

# Read the JSON file for MRI labels
with open(txt_path, 'r') as json_reader:
    mri_label = json.load(json_reader)

# Get a list of all subjects
all_subject = sorted(glob.glob(img_path + '/**/*.nii.gz'))

# Split the subjects into training and testing based on the flag
if is_train:
    subject_list = all_subject[:int(len(all_subject)*0.95)]  # 95%
else:
    subject_list = all_subject[int(len(all_subject)*0.95):]  # Remaining 5%

# Create a directory for subjects not in label_keys
not_in_label_keys_dir = '/scratch/slurm-user3/changsun/data/not_in_label_keys/'
os.makedirs(not_in_label_keys_dir, exist_ok=True)

# Iterate over each subject
# Iterate over each subject
for subject in subject_list:
    patient_ID = subject.split('/')[-2]
    patient_ID_key = patient_ID[:-11]
    #print(patient_ID_key)
    # Define the destination directory for this subject
    destination_dir = os.path.join(not_in_label_keys_dir, os.path.basename(os.path.dirname(subject)))
    #print(mri_label.keys())
    # Check if patient_ID_key is not in mri_label and if the destination directory does not exist
    if patient_ID_key not in mri_label.keys() and not os.path.exists(destination_dir):
        # Move the folder to the not_in_label_keys directory
        shutil.move(os.path.dirname(subject), destination_dir)


# Note: This code will move the entire directory for each subject not found in label_keys