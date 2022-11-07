import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import nibabel as nib

path_test = "/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/imagesTs"
path_brno = "/mnt/CRAI-NAS/all/martinsr/NNunet/data/test_predictions/imagesTs_brno"

test_files = glob.glob(path_test + "/*.nii.gz")
brno_files = glob.glob(path_brno + "/*.nii.gz")

all_positive_values_test = []
all_positive_values_brno = []


for i in range(len(test_files)):
    test = nib.load(test_files[i]).get_fdata()

    test_positive_values = test[test > 0]
    test_positive_values = test_positive_values.flatten()

    all_positive_values_test.append(np.mean(test_positive_values))


for j in range(len(brno_files)):
    brno = nib.load(brno_files[j]).get_fdata()

    brno_positive_values = brno[brno > 0]

    brno_positive_values = brno_positive_values.flatten()

    all_positive_values_brno.append(np.mean(brno_positive_values))


# get p value for overlap of unbalanced sample sizes distributions
from scipy.stats import mannwhitneyu
stat, p = mannwhitneyu(all_positive_values_test, all_positive_values_brno)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')


# plot histogram overlap
plt.figure(figsize=(10, 10))
# kde plot
sns.kdeplot(all_positive_values_test, shade=True, color="r", label="Internal test set", alpha=0.5)
sns.kdeplot(all_positive_values_brno, shade=True, color="b", label="External test set", alpha=0.5)
plt.legend()
plt.title(f"Distribution p={p}", fontsize=25)
plt.xlabel("Mean intensity", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
plt.savefig("histogram_overlap_brno.png")