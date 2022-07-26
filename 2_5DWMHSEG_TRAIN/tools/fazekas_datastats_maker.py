import sys
from skimage.measure import label, regionprops
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pickle
from skimage.morphology import remove_small_objects
sys.path.insert(1, '../')
import utils



def make_fazekas_data(dataset, type:str='train'):
    output = []
    for subject in dataset:
        sx, sy, sz = nib.load(subject['image']).header.get_zooms()
        mask_label = nib.load(subject['image']).get_fdata() # Mask loads in as 0-1
        mask_label[mask_label > 0.89] = 1
        mask_label[mask_label <= 0.89] = 0
        mask_label = mask_label.astype('int')
        mask_label_ = label(mask_label, connectivity=2)
        mask_prop = regionprops(mask_label_)
        lesions = []
        lesions_centroid_mm = []
        lesions_cleaned = []
        lesions_centroid_mm_cleaned = []
        to_output = {'labelpath': [], 'label': [], 'voxel_res': [], 'lesions': [], 'lesion_hist': [], 'lesions_centroid_mm': [], 'lesions_centroid_mm_cleaned': [], 'lesions_cleaned': [], 'lesion_hist_cleaned': []}

        #DatasetWMH211018_v2/fazekas3_220107/train_faz
        #DatasetWMH211018_v2/fazekas3_220107/val_faz
        
        mask_label_cleaned = mask_label.astype('int').astype('bool')
        mask_label_cleaned = remove_small_objects(mask_label_cleaned, min_size = 5, connectivity = 2, in_place = True)
        mask_label_cleaned = remove_small_objects(mask_label_cleaned, min_size = 5, connectivity = 2, in_place = True)
        mask_label_cleaned = mask_label_cleaned.astype('int')
        mask_label_cleaned_ = label(mask_label_cleaned, connectivity=2)
        mask_label_cleaned_prop = regionprops(mask_label_cleaned_)


        # Center of the volume
        center = np.array([mask_label_.shape[0]/2, mask_label_.shape[1]/2, mask_label_.shape[2]/2])
        for prop_pred in mask_prop:
            volume = prop_pred['area']*subject["voxel_res"]
            cx, cy, cz = prop_pred.centroid
            ccx = round((cx - center[0]) * sx, 2)
            ccy = round((cy - center[1]) * sy, 2)
            ccz = round((cz - center[2]) * sz, 2)
            lesions.append(volume)
            lesions_centroid_mm.append([ccx, ccy, ccz])
            
        for prop_pred in mask_label_cleaned_prop:
            volume = prop_pred['area']*subject["voxel_res"]
            cx, cy, cz = prop_pred.centroid
            ccx = round((cx - center[0]) * sx, 2)
            ccy = round((cy - center[1]) * sy, 2)
            ccz = round((cz - center[2]) * sz, 2)
            lesions_cleaned.append(volume)
            lesions_centroid_mm_cleaned.append([ccx, ccy, ccz])
        
        lesions = np.array(lesions)
        to_output['lesions'] = lesions.copy()
        to_output['lesions_centroid_mm'] = np.array(lesions_centroid_mm)
        lesions[lesions > 9990] = 9990
        hist, bin = np.histogram(lesions, bins = np.arange(0, 10000, 30))

        to_output['labelpath'] = subject['image']
        to_output['voxel_res'] = subject['voxel_res']
        to_output['label'] = subject['label']
        to_output['lesion_hist'] = hist

        # Cleaned
        lesions_cleaned = np.array(lesions_cleaned)
        to_output['lesions_cleaned'] = lesions_cleaned.copy()
        to_output['lesions_centroid_mm_cleaned'] = np.array(lesions_centroid_mm_cleaned)
        lesions_cleaned[lesions_cleaned > 9990] = 9990
        hist_cleaned, bin = np.histogram(lesions_cleaned, bins = np.arange(0, 10000, 30))
        to_output['lesion_hist_cleaned'] = hist_cleaned

        output.append(to_output)
        
    with open(f'../dataanalysis/fazekas_data_{type}.pickle', 'wb') as handle:
        pickle.dump(output, handle)

    

trainfolder = "/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/train"
valfolder = "/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/val"
trainfolderfaz = "/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/fazekas3_220107/train_faz"
valfolderfaz = "/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/fazekas3_220107/val_faz"
# trainfolder = r'C:\Users\Gimpe\Google Drive\Master -Signal_processingWORK\Masteroppgave\Main_code\data\train'
# valfolder = r"C:\Users\Gimpe\Google Drive\Master -Signal_processingWORK\Masteroppgave\Main_code\data\val"


# Add fazekas 3 from custom dataset
train_files, val_files = utils.dataprocesser_fazekas(trainfolder, valfolder, dict_sep = True, filename = 'F_seg.nii.gz') # [{"image": image, "label": label, "voxel_res":
train_files_faz, val_files_faz = utils.dataprocesser_fazekas(trainfolderfaz, valfolderfaz, hardfazval = 3, filename = 'F_seg.nii.gz', dict_sep = True) # [{"image": image, "label": label, "voxel_res":

# Append it to the original
train_files = train_files + train_files_faz
val_files = val_files + val_files_faz

make_fazekas_data(train_files, type = 'train')
make_fazekas_data(val_files, type = 'val')
        

# with open('../dataanalysis/fazekas_data_val.pickle', 'rb') as handle:
#     val = pickle.load(handle)

# ll = np.array([])
# for subject in val:
#     ll = np.append(ll, subject['lesions'])

# plt.hist(ll.flatten(), bins = 100)
# plt.show()