import nibabel as nib
import glob
from tqdm import tqdm

path_to_all = glob.glob(
    "/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/Projects/Lymedisease_control/*.nii.gz"
)
save_to = (
    "/mnt/CRAI-NAS/all/martinsr/NNunet/data/test_predictions/imagesTr_control_edited/"
)

for path in tqdm(path_to_all):
    img = nib.load(path)
    # canonicol
    img = nib.as_closest_canonical(img)
    # save
    save_path = save_to + path.split("/")[-1]
    nib.save(img, save_path)
    # print
    print("Saved: ", save_path)
