
import glob
# simpleitk
import SimpleITK as sitk
import numpy as np
target_path = "/mnt/CRAI-NAS/all/martinsr/NNunet/results/predictions_epoch=550-dice_mean=77_44_task=15_fold=0_tta/*"


wmh = {}
# calculate the total wmh volume in ml
for i in glob.glob(target_path):
    img = sitk.ReadImage(i)
    spacing = img.GetSpacing()
    img = sitk.GetArrayFromImage(img)
    total_wmh = np.sum(img)
    # get total wmh in ml
    total_wmh = total_wmh * 0.001
    # calculate volume
    total_wmh = total_wmh * spacing[0] * spacing[1] * spacing[2]
    id = i.split("/")[-1].split(".")[0]
    print(id, spacing)
    wmh[id] = round(total_wmh,3)


# save as json
import json
with open(f'{target_path.replace("*", "")}wmh_total.json', 'w') as fp:
    json.dump(wmh, fp, indent=4)