#%%
import matplotlib.pyplot as plt
import utils
import nibabel as nib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
from sklearn.metrics import classification_report
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy
from scipy.stats import shapiro
from skimage.measure import label, regionprops
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
import pickle
import pandas as pd
import graphviz
from sklearn.preprocessing import PowerTransformer
from rich.progress import track
from sklearn.manifold import LocallyLinearEmbedding
np.random.seed(0)
from skimage.morphology import remove_small_objects

#%%



def get_results(ypred:np.array, y:np.array, ypaths:np.array, targetlabel:int) -> np.array:
    idx = np.where(ypred == targetlabel)
    y_true = y[idx]
    ylabels = ypaths[idx]
    output = {}
    for i, path in enumerate(ylabels):
        output[path] = y_true[i]
    return output



class MultiVariateClassifier(object):
    """Multivariate gaussian classifier"""
    def __init__(self, classification_type = 'tree', mode = 'custom', use_ensamble = False, n_estimators = 50, learning_rate = 1):
        self.classification_type = classification_type
        self.use_ensamble = use_ensamble
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        # self.clf = AdaBoostClassifier(
        #     DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200
        #                   )               

        if mode == 'custom':
            self.datasets = {'train': {'features': [],  'labels': [], 'labelpath': [], 'lesions': []}, 'val': {'features': [], 'labels': [], 'labelpath': [], 'lesions': []}}
            datasets = ['train', 'val']
            for data_type in datasets:
                with open(f'dataanalysis/fazekas_data_{data_type}_pred.pickle', 'rb') as handle:
                    data = pickle.load(handle)
                for i, subject in enumerate(data):
                    subject['lesions'][subject['lesions'] > 9990] = 9990
                    mean = np.mean(subject['lesions_cleaned'])
                    std = np.std(subject['lesions_cleaned'])
                    long = len(subject['lesions_cleaned'])
                    centroids = np.array(subject['lesions_centroid_mm_cleaned'])
                    centroidsx = np.mean(centroids[:, 0])
                    centroidsy = np.mean(centroids[:, 1])
                    centroidsz = np.mean(centroids[:, 2])
                    centroidsx_std = np.std(centroids[:, 0])
                    centroidsy_std = np.std(centroids[:, 1])
                    centroidsz_std = np.std(centroids[:, 2])
                    feat = np.array([mean, std, long, centroidsx, centroidsy, centroidsz, centroidsx_std, centroidsy_std, centroidsz_std])[None, :]
                    if i == 0:
                        # self.datasets[f'{data_type}']['features'] = subject['lesion_hist'][None, :]
                        self.datasets[f'{data_type}']['features'] = feat
                        self.datasets[f'{data_type}']['labels'].append(subject['label'])
                        self.datasets[f'{data_type}']['labelpath'].append(subject['labelpath'])
                    else:
                        # self.datasets[f'{data_type}']['features'] = np.concatenate((self.datasets[f'{data_type}']['features'], subject['lesion_hist'][None, :]), axis = 0)

                        self.datasets[f'{data_type}']['features'] = np.concatenate((self.datasets[f'{data_type}']['features'], feat), axis = 0)
                        self.datasets[f'{data_type}']['labels'].append(subject['label'])
                        self.datasets[f'{data_type}']['labelpath'].append(subject['labelpath'])

    def test_features_normality(self):
        self.feature_rejections = []
        for feature in self.datasets['train']['features']:
            feature_data = self.datasets['train']['features'][feature]
            stat, p = shapiro(feature_data)
            # interpret
            alpha = 0.05
            if p > alpha:
                self.feature_rejections.append(True)
                print('Feature looks Gaussian (fail to reject H0); ', feature)
            else:
                self.feature_rejections.append(False)
                print('Feature does not look Gaussian (reject H0); ', feature)
        return self.feature_rejections


    def make_model(self, not_training = True):
        if self.classification_type == 'tree':
            if not_training:
                self.clf = AdaBoostClassifier(n_estimators=self.n_estimators, random_state=0, learning_rate = self.learning_rate)
            X = self.datasets['train']['features']
            y = self.datasets['train']['labels']
            self.clf.fit(X, y)


    def evaluate(self, loadmodel = False, verbatim = True):
        if verbatim:
            print('Running evaluation...')
        if loadmodel:
            self.clf = pickle.load(open(loadmodel, 'rb'))
            print('Model loaded')
        X = self.datasets['val']['features']
        y = np.array(self.datasets['val']['labels'])
        image_paths = np.array(self.datasets['val']['labelpath'])
        predictions = self.clf.predict(X)


        target_names = ['fazekas 0', 'fazekas 1', 'fazekas 2', 'fazekas 3']

        if not self.use_ensamble:
            dot_data = tree.export_graphviz(self.clf, 
                                feature_names=[feature for feature in self.datasets['train']['features']],  
                                class_names=target_names,  
                                filled=True, rounded=True,  
                                special_characters=True)
            graph = graphviz.Source(dot_data)
            graph.render(filename = 'classification_tree')
        if verbatim:
            print(get_results(predictions, y, image_paths, 2))
            print(classification_report(y, np.array(predictions).ravel(), target_names=target_names))
        output = classification_report(y, np.array(predictions).ravel(), target_names=target_names, output_dict=True, zero_division=1)

        return output
    

    def train_model_hyperparams(self, max_n_estimators = 100, max_learning_rate = 10, learnin_rate_step = 0.01):
        best_score = 0
        best_n_estimators = 0
        for n_est in track(range(2, max_n_estimators)):
            for learning_rate in np.arange(0.01, max_learning_rate, learnin_rate_step):
                learning_rate = round(learning_rate, 2)
                self.clf = AdaBoostClassifier(n_estimators=n_est, random_state=0, learning_rate = learning_rate)
                # self.clf = AdaBoostClassifier(
                #             DecisionTreeClassifier(max_depth=1, min_samples_leaf=1),
                #             n_estimators=10000,
                            # learning_rate=learning_rate)
                self.make_model(not_training = False)
                results = self.evaluate(verbatim = False)

                if results['macro avg']['f1-score'] > best_score:
                    best_score = results['macro avg']['f1-score']
                    best_n_estimators = n_est
                    best_learningrate = learning_rate
                    pickle.dump(self.clf, open('saved_models/classification_model.pickle', 'wb'))
                    print('Best score; ',best_score, 'Best n_estimators: ', best_n_estimators, 'Best learning rate: ', best_learningrate)
        print('END Best score; ',best_score, 'Best n_estimators: ', best_n_estimators, 'Best learning rate: ', best_learningrate)


    def predict(self, path, modelpath = 'saved_models/adaboost_simple.pickle'):
        print('Running predictions...')
        dataset = utils.dataprocesser_pred_fazekas(path)
        self.clf = pickle.load(open(modelpath, 'rb'))
        print('Model loaded')
        for subject in track(dataset):
            sx, sy, sz = nib.load(subject['image']).header.get_zooms()
            mask_label = nib.load(subject['image']).get_fdata() # Mask loads in as 0-1
            mask_label[mask_label > 0.89] = 1
            mask_label[mask_label <= 0.89] = 0
            lesions_cleaned = []
            lesions_centroid_mm_cleaned = []
            to_output = {'labelpath': [], 'label': [], 'voxel_res': [], 'lesions': [], 'lesion_hist': [], 'lesions_centroid_mm': [], 'lesions_centroid_mm_cleaned': [], 'lesions_cleaned': [], 'lesion_hist_cleaned': []}

            vol = sx*sy*sz
            mask_label_cleaned = mask_label.astype('int').astype('bool')
            mask_label_cleaned = remove_small_objects(mask_label_cleaned, min_size = round(5/vol), connectivity = 2, in_place = True)
            mask_label_cleaned = remove_small_objects(mask_label_cleaned, min_size = round(5/vol), connectivity = 2, in_place = True)
            mask_label_cleaned = mask_label_cleaned.astype('int')
            mask_label_cleaned_ = label(mask_label_cleaned, connectivity=2)
            mask_label_cleaned_prop = regionprops(mask_label_cleaned_)
            # # Center of the volume
            center = np.array([mask_label.shape[0]/2, mask_label.shape[1]/2, mask_label.shape[2]/2])
            # for prop_pred in mask_prop:
            #     volume = prop_pred['area']*subject["voxel_res"]
            #     cx, cy, cz = prop_pred.centroid
            #     ccx = round((cx - center[0]) * sx, 2)
            #     ccy = round((cy - center[1]) * sy, 2)
            #     ccz = round((cz - center[2]) * sz, 2)
            #     lesions.append(volume)
            #     lesions_centroid_mm.append([ccx, ccy, ccz])
            for prop_pred in mask_label_cleaned_prop:
                volume = prop_pred['area'].copy()*vol
                cx, cy, cz = prop_pred.centroid
                ccx = round((cx - center[0]) * sx, 2)
                ccy = round((cy - center[1]) * sy, 2)
                ccz = round((cz - center[2]) * sz, 2)
                lesions_cleaned.append(volume)
                lesions_centroid_mm_cleaned.append([ccx, ccy, ccz])
            
            # lesions = np.array(lesions)
            # to_output['lesions'] = lesions.copy()
            # to_output['lesions_centroid_mm'] = np.array(lesions_centroid_mm)
            # lesions[lesions > 9990] = 9990
            # hist, bin = np.histogram(lesions, bins = np.arange(0, 10000, 30))

            # to_output['labelpath'] = subject['image']
            # to_output['voxel_res'] = subject['voxel_res']
            # to_output['label'] = subject['label']
            # to_output['lesion_hist'] = hist

            # Cleaned
            lesions_cleaned = np.array(lesions_cleaned)
            to_output['lesions_cleaned'] = lesions_cleaned.copy()
            to_output['lesions_centroid_mm_cleaned'] = np.array(lesions_centroid_mm_cleaned)
            lesions_cleaned[lesions_cleaned > 9990] = 9990
            hist_cleaned, bin = np.histogram(lesions_cleaned, bins = np.arange(0, 10000, 30))
            to_output['lesion_hist_cleaned'] = hist_cleaned

            mean = np.mean(to_output['lesions_cleaned'])
            std = np.std(to_output['lesions_cleaned'])
            long = len(to_output['lesions_cleaned'])
            centroids = np.array(to_output['lesions_centroid_mm_cleaned'])
            centroidsx = np.mean(centroids[:, 0])
            centroidsy = np.mean(centroids[:, 1])
            centroidsz = np.mean(centroids[:, 2])
            centroidsx_std = np.std(centroids[:, 0])
            centroidsy_std = np.std(centroids[:, 1])
            centroidsz_std = np.std(centroids[:, 2])
            X = np.array([mean, std, long, centroidsx, centroidsy, centroidsz, centroidsx_std, centroidsy_std, centroidsz_std])[None, :]
            predictions = self.clf.predict(X)[0]
            print('Prediction using: ', subject['image'])
            fazekas_output_path = subject['image'].replace('F_seg.nii.gz', 'fazekas_scale.txt')
            with open(fazekas_output_path, 'w') as f:
                f.write(f"Fazekas: {predictions}")
    

    def roc_analysis(self, classes = 4):
        if self.classification_type == 'tree':
            pass
        else:
            y = label_binarize(self.datasets['val']['labels'], classes=np.arange(0, classes))
            n_classes = y.shape[1]
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            # Compute ROC curve and ROC area for each class
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y[:, i], self.predictions_probability[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], thresholds = roc_curve(y.ravel(), self.predictions_probability.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            lw = 2
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            
            # Finally average it and compute AUC
            mean_tpr /= n_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr["micro"],
                tpr["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr["macro"],
                tpr["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )

            for i in range(n_classes):
                plt.plot(
                    fpr[i],
                    tpr[i],
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Multiclass Receiver operating characteristic")
            plt.legend(loc="lower right")
            plt.show()



#%%


# for i in range(1, 6):
# print('ensamble: ', i)
classifer = MultiVariateClassifier(classification_type = 'tree', mode = 'custom', use_ensamble = True, n_estimators = 47, learning_rate = 4.39)

# classifer = MultiVariateClassifier(classification_type = 'tree', mode = 'None', max_depth = 2, use_ensamble=True, n_estimators= 7, learning_rate = 1)
# classifer.add_features(N = 3) # Best N = 3
# classifer.add_dataset(data_train, set_type = 'train')
# classifer.add_dataset(data_val, set_type = 'val')
# classifer.add_features(feature_type = 'pca3')
# classifer.add_features(feature_type = 'entropy')
# classifer.add_features(feature_type = 'size')
# classifer.add_features(feature_type = 'lesions')
# classifer.test_features_normality()
# classifer.train_model_hyperparams() #Best score;  0.7660714285714285 39 Best learning rate:  3.229999999999999
# classifer.make_model()
# classifer.evaluate()
classifer.evaluate(loadmodel = 'saved_models/classification_model.pickle')

# classifer.predict(r'C:\Users\Gimpe\Google Drive\Master -Signal_processingWORK\Masteroppgave\Main_code\data\val', modelpath = 'saved_models/classification_model.pickle')

# %%
# classifer.evaluate()
# classifer.roc_analysis()
# %%
