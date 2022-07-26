from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import glob


path = "/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/test_sample"
path_fazekas_gt = "/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/test_sample/Fazekas.txt"

fazekas_label_df = pd.read_csv(path_fazekas_gt, sep=",", header=None)
fazekas_predicted = {}

print(fazekas_label_df)


subjects = glob.glob(path+'/*')
subjects = [x for x in subjects if 'scoring.txt' not in x and 'Fazekas.txt' not in x and 'leasion_counting' not in x]


predicted = []
labels = []

for subject in subjects:
    pred_fazekas = int(np.genfromtxt(subject+'/fazekas_scale.txt')[1])
    subject_id = subject.split('/')[-1]
    predicted.append(pred_fazekas)
    label_fazekas = fazekas_label_df[0].values
    label_fazekas_number = fazekas_label_df[1].values
    idx = np.where(label_fazekas == subject_id)
    label_val = label_fazekas_number[idx][0]
    labels.append(label_val)
    print(subject_id, (pred_fazekas), [label_val])

print(classification_report(labels, predicted))