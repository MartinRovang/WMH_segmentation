import SimpleITK as sitk
import seaborn as sns
import glob
import numpy as np
import matplotlib.pyplot as plt

path = "/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/labelsTs/*.nii.gz"
all_wmh = []


for file in glob.glob(path):
    img = sitk.ReadImage(file)
    wmh = sitk.GetArrayFromImage(img)
    resolution = img.GetSpacing()
    all_wmh.append(np.sum(wmh)*resolution[0]*resolution[1]*resolution[2]*0.001)


sns.histplot(all_wmh, kde=False)
plt.xlabel("Volume (ml)")
plt.ylabel("Number of participants")
plt.savefig("wmh_distribution_plot.svg")
plt.savefig("wmh_distribution_plot.tif")
plt.savefig("wmh_distribution_plot.jpg")

# median and interquantiles
print(f"{np.median(all_wmh)} ({np.quantile(all_wmh, 0.25)}, {np.quantile(all_wmh, 0.75)})")
# total range
print(f"Range: {np.min(all_wmh)}, {np.max(all_wmh)}")
