#%%
import matplotlib.pyplot as plt
import utils
import nibabel as nib
import monai
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    EnsureTyped,
    Resized,
    NormalizeIntensityd
)
from scipy.stats import shapiro
from scipy.stats import entropy
from scipy.stats import multivariate_normal
from sklearn.metrics import classification_report
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy
from scipy.stats import shapiro
from skimage.measure import label, regionprops
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import sklearn
import pickle
import pandas as pd
import graphviz
from sklearn.preprocessing import PowerTransformer
from rich.progress import track
np.random.seed(1)

# trainfolder = "/mnt/CRAI-NAS/all/martinsr/dataset_OK/train"
# valfolder = "/mnt/CRAI-NAS/all/martinsr/dataset_OK/val"
# trainfolder = r"C:\Users\Gimpe\Google Drive\Master -Signal_processingWORK\Masteroppgave\Main_code\data\train"
# valfolder = r"C:\Users\Gimpe\Google Drive\Master -Signal_processingWORK\Masteroppgave\Main_code\data\val"
# train_files, val_files = utils.dataprocesser_fazekas(trainfolder, valfolder, dict_sep = True)

class RemoveEigthSlices:
    def __init__(self, keys):
        self.keys = keys


    def __call__(self, image_dict):
        # Remove slices with little information

        image = image_dict[self.keys[0]]
        image[image > 0.89] = 1
        image[image <= 0.89] = 0
        image = image[image.shape[0]//12:-image.shape[0]//12, image.shape[1]//12:-image.shape[1]//12, image.shape[2]//12:-image.shape[2]//12]
        image_dict[self.keys[0]] = image

        return image_dict



load_fazekas = Compose(
    [
        LoadImaged(keys=["image"]),
        RemoveEigthSlices(keys=["image"]),
        AddChanneld(keys=["image"]),
        Resized(keys=["image"], spatial_size = (64, 64, 64), mode = 'nearest'),
        # NormalizeIntensityd(keys=["image"]),
        EnsureTyped(keys=["image"], data_type = 'numpy')
    ]
)

# train_ds = monai.data.Dataset(data = train_files, transform = load_fazekas)
# val_ds = monai.data.Dataset(data = val_files, transform = load_fazekas)



def make_data(ds):
    for j, i in enumerate(ds):
        img = i['image'].flatten()
        if j > 0:
            X = np.concatenate((X, img[None, :]), axis=0)
            labels.append(i['label'])
            voxel_res.append(i['voxel_res'])
        else:
            X = img[None, :]
            labels = [i['label']]
            voxel_res = [i['voxel_res']]
    return X, labels, voxel_res

# X, labels, voxel_res = make_data(train_ds)
# X_val, X_labels, voxel_res_val = make_data(val_ds)
# np.save('train', data = X, labels = labels, voxel = voxel_res)
# np.save('val', data = X_val, labels = X_labels, voxel = voxel_res_val)
# data_train = {'data': X, 'labels': labels}
# data_val = {'data': X_val, 'labels': X_labels}


# train_load = np.load('dataanalysis/train.npz')
# val_load = np.load('dataanalysis/val.npz')

# X_train, train_label, voxel_train = train_load['data'], train_load['labels'], train_load['voxel']
# X_val, val_label, voxel_val = val_load['data'], val_load['labels'], val_load['voxel']

# # X_train = X_train
# # X_val = X_val

# data_train = {'data': X_train, 'labels': train_label}
# data_val = {'data': X_val, 'labels': val_label}

#%%

class MultiVariateClassifier(object):
    best_score = 0
    """Multivariate gaussian classifier"""
    def __init__(self, classification_type = 'tree', mode = 'custom', max_depth = 50, use_ensamble = False, n_estimators = 50, learning_rate = 1):
        self.classification_type = classification_type
        self.use_ensamble = use_ensamble
        self.n_estimators = n_estimators
        self.scaler = StandardScaler()
        if classification_type == 'tree':
            self.clf = tree.DecisionTreeClassifier(random_state=0, max_depth=max_depth, criterion='gini', max_features="auto") # Current best with 2 depth
            if self.use_ensamble:
                self.clf = AdaBoostClassifier(n_estimators=self.n_estimators, random_state=0, learning_rate = learning_rate)
        self.mode = mode # Legacy data if not custom
        self.datasets = {}
        self.datasets_rad = {}
        if mode == 'custom':
            self.datasets = {'train': {'features':{}}, 'val': {'features':{}}}
            self.datasets_rad = {}
            with open('dataanalysis/train_radiometrics.pickle', 'rb') as handle:
                data = dict(pickle.load(handle))
                self.datasets_rad['train'] = data
                for subject in self.datasets_rad['train']:
                    for feature in self.datasets_rad['train'][subject]:
                        feature_value = self.datasets_rad['train'][subject][feature]
                        if feature in self.datasets['train']['features']:
                            self.datasets['train']['features'][feature].append(feature_value)
                        else:
                            self.datasets['train']['features'][feature] = [feature_value]
            with open('dataanalysis/val_radiometrics.pickle', 'rb') as handle:
                data = dict(pickle.load(handle))
                self.datasets_rad['val'] = data
                for subject in self.datasets_rad['val']:
                    for feature in self.datasets_rad['val'][subject]:
                        feature_value = self.datasets_rad['val'][subject][feature]
                        if feature in self.datasets['val']['features']:
                            self.datasets['val']['features'][feature].append(feature_value)
                        else:
                            self.datasets['val']['features'][feature] = [feature_value]
        

    def add_dataset(self, x:dict, set_type:str = 'train'):
        if set_type == 'train' or set_type == 'val':
            self.datasets[set_type] = {'data': x['data'], 'labels': x['labels']}
        else:
            self.datasets[set_type] = {'data': x['data']}
        print(f'Dataset {set_type} added!')

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
    

    def add_features(self, feature_type:str = 'entropy', N = 4, loadscaler = False):
        if self.mode == 'custom':
            self.analyze_best_features(N, loadscaler = loadscaler)
        else:
            for dataset_type in self.datasets:
                data = self.datasets[dataset_type]['data']
                if 'features' not in self.datasets[dataset_type]:
                    self.datasets[dataset_type]['features'] = {}
                if feature_type == 'entropy':
                    self.datasets[dataset_type]['features']['entropy'] = entropy(data, axis = 1)
                if feature_type == 'size':
                    self.datasets[dataset_type]['features']['totalsize'] = data.sum(1)
                
                if feature_type == 'pca3':
                    data = self.datasets[dataset_type]['data']
                    mu = np.mean(data, axis = 1)[:, None]
                    std = np.std(data, axis = 1)[:, None]
                    # Standardize
                    data_normed = (data - mu)/std
                    if dataset_type == 'train':
                        self.pca = PCA(n_components=3).fit(data_normed)
                    pca_components = self.pca.transform(data_normed)
                    pca3 = pca_components[:, 2]
                    # if dataset_type == 'train':
                        # bc = PowerTransformer(method="yeo-johnson")
                        # self.bc = bc.fit(pca3[:, None])
                    # pca3_trans = self.bc.transform(pca3[:, None]).flatten()
                    self.datasets[dataset_type]['features']['pca3'] = pca3

            
            if feature_type == 'lesions':
                al_total_lesions = np.zeros(data.shape[0])
                al_mean_lesions = np.zeros(data.shape[0])
                al_std_lesions = np.zeros(data.shape[0])
                for index in range(0, data.shape[0]):
                    vol = data[index, :].reshape((64, 64, 64))
                    # label image regions
                    label_image = label(vol, connectivity=2, background=0)
                    lesionprops = regionprops(label_image)
                    total_lesions = len(np.unique(label_image))
                    all_lesions = [prop.area*voxel_train[index] for prop in lesionprops] 
                    mean_lesions = np.mean(all_lesions)
                    std_lesions = np.std(all_lesions)

                    al_total_lesions[index] = total_lesions
                    al_mean_lesions[index] = mean_lesions
                    al_std_lesions[index] = std_lesions

                self.datasets[dataset_type]['features']['Total_lesions'] = al_total_lesions
                self.datasets[dataset_type]['features']['Mean_lesions'] = al_mean_lesions
                self.datasets[dataset_type]['features']['std_lesions'] = al_std_lesions
    

    def make_model(self):
        if self.classification_type == 'tree':
            for j, feature in enumerate(self.datasets['train']['features']):
                if j > 0:
                    X = np.concatenate((X, np.array(self.datasets['train']['features'][feature])[:, None]), axis = 1)
                else:
                    X = np.array(self.datasets['train']['features'][feature])[:, None]
            
            self.clf.fit(X, np.array(self.datasets['train']['labels']).ravel())

        else:
            self.MU = {}
            self.COV = {}

            labels = np.array(self.datasets['train']['labels'], dtype = 'int')
            idx0 = np.where(labels == 0)
            idx1 = np.where(labels == 1)
            idx2 = np.where(labels == 2)
            idx3 = np.where(labels == 3)
            classes_indices = [idx0, idx1, idx2, idx3]

            for i_class, index in enumerate(classes_indices):
                COV_temp = []
                self.MU[f'class_{i_class}'] = []
                for feature in self.datasets['train']['features']:
                    if len(index[0]) > 0:
                        self.MU[f'class_{i_class}'].append(np.mean(np.array(self.datasets['train']['features'][feature])[index]))
                        COV_temp.append(np.array(self.datasets['train']['features'][feature])[index])
                if len(COV_temp) > 0:
                    self.COV[f'class_{i_class}'] = np.cov(COV_temp)
        

    def evaluate(self, verbatim = False, loadmodel = False):
        if loadmodel:
            self.clf = pickle.load(open(f'{loadmodel}', 'rb'))
        if verbatim:
            print('Running evaluation...')
        if self.classification_type == 'tree':
            for j, feature in enumerate(self.datasets['val']['features']):
                if j > 0:
                    X = np.concatenate((X, np.array(self.datasets['val']['features'][feature])[:, None]), axis = 1)
                else:
                    X = np.array(self.datasets['val']['features'][feature])[:, None]
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
                print(classification_report(np.array(self.datasets['val']['labels']).ravel(), np.array(predictions).ravel(), target_names=target_names))
            output = classification_report(np.array(self.datasets['val']['labels']).ravel(), np.array(predictions).ravel(), target_names=target_names, output_dict=True, zero_division=1)
            return output
        else:
            predictions = []
            N_classes = len(np.unique(self.datasets['train']['labels']))
            N_samples = len(self.datasets['val']['labels'])
            self.predictions_probability = np.zeros((N_samples, N_classes))
            for data_index in range(0, N_samples):
                # Prediction
                predictions_intermidate = []
                for class_name in self.MU:
                    muu = self.MU[class_name]
                    if len(muu) > 0:
                        coo = self.COV[class_name]
                        input = []
                        for feature in self.datasets['val']['features']:
                            input.append(self.datasets['val']['features'][feature][data_index])

                        y = multivariate_normal.pdf(input, muu, cov=coo)
                        predictions_intermidate.append(y)

                predictions.append(np.argmax(predictions_intermidate))
                self.predictions_probability[data_index, :] = predictions_intermidate
            target_names = ['fazekas 0', 'fazekas 1', 'fazekas 2', 'fazekas 3']

            output = classification_report(self.datasets['val']['labels'].ravel(), predictions.ravel(), target_names=target_names, output_dict=True)
            print(output)
            return output


    def train_model_hyperparams(self, max_n_estimators = 100, max_learning_rate = 10, learnin_rate_step = 0.01):
        best_n_estimators = 0
        best_learningrate = 0

        for n_est in track(range(2, max_n_estimators)):
            for learning_rate in np.arange(0.01, max_learning_rate, learnin_rate_step):
                learning_rate = round(learning_rate, 2)
                self.clf = AdaBoostClassifier(n_estimators=n_est, random_state=0, learning_rate = learning_rate)
                self.make_model()
                results = self.evaluate(verbatim = False)
                if results['macro avg']['f1-score'] > MultiVariateClassifier.best_score:
                    MultiVariateClassifier.best_score = results['macro avg']['f1-score']
                    best_n_estimators = n_est
                    best_learningrate = learning_rate
                    pickle.dump(self.clf, open('saved_models/classification_model.pickle', 'wb'))
                    print('Best score; ',MultiVariateClassifier.best_score, 'Best n_estimators: ', best_n_estimators, 'Best learning rate: ', best_learningrate)


        print('END Best score; ',MultiVariateClassifier.best_score, 'Best n_estimators: ', best_n_estimators, 'Best learning rate: ', best_learningrate)


    def predict(self):
        print('Running predictions...')
        predictions = []
        N_samples = len(self.datasets['pred']['data'])
        for data_index in range(0, N_samples):
            # Prediction
            predictions_intermidate = []
            for class_name in self.MU:
                muu = self.MU[class_name]
                if len(muu) > 0:
                    coo = self.COV[class_name]
                    input = []
                    for feature in self.datasets['pred']['features']:
                        input.append(self.datasets['pred']['features'][feature][data_index])
                    y = multivariate_normal.pdf(input, muu, cov=coo)
                    predictions_intermidate.append(y)

            predictions.append(np.argmax(predictions_intermidate))
        return predictions
    

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

    def analyze_best_features(self, N = 4, loadscaler = False, loadpca = False):
        #https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
        #https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
        
        remove_list = ['diagnostics_Configuration_Settings', 'diagnostics_Versions_PyRadiomics', 'diagnostics_Versions_Numpy', 'diagnostics_Versions_SimpleITK', 'diagnostics_Versions_PyWavelet', 'diagnostics_Versions_Python', 'diagnostics_Configuration_EnabledImageTypes', 'diagnostics_Image-original_Hash', 'diagnostics_Image-original_Dimensionality', 'diagnostics_Mask-original_Hash', 'diagnostics_Image-original_Mean', 'diagnostics_Mask-original_VoxelNum', 'diagnostics_Mask-original_VolumeNum', 'Image', 'diagnostics_Image-original_Minimum', 'diagnostics_Image-original_Maximum', 'diagnostics_Image-original_Spacing', 'diagnostics_Image-original_Size', 'diagnostics_Mask-original_Size', 'diagnostics_Mask-original_Spacing', 'diagnostics_Mask-original_BoundingBox', 'diagnostics_Versions_PyRadiomics', 'diagnostics_Versions_Numpy']
        columns=[key for key in self.datasets['train']['features'] if key not in remove_list and type(self.datasets['train']['features'][key][0]) != tuple]
        df_train = pd.DataFrame(self.datasets['train']['features'], columns=columns)
        df_val = pd.DataFrame(self.datasets['val']['features'], columns=columns)
        # Separating out the features
        df_fazekas_train = df_train[['Fazekas']]
        df_fazekas_val = df_val[['Fazekas']]
        df_train = df_train.drop('Fazekas', axis = 1)
        df_val = df_val.drop('Fazekas', axis = 1)


        x_train = df_train.loc[:, df_train.columns].values
        x_val = df_val.loc[:, df_val.columns].values

        # Standardizing the features
        x_train = self.scaler.fit_transform(x_train)
        pca = PCA(n_components=N)
        principalComponents_train = pca.fit_transform(x_train)
        if loadscaler:
            self.scaler = pickle.load(open(f'{loadscaler}', 'rb'))
            pca = pickle.load(open(f'{loadpca}', 'rb'))
        else:
            pickle.dump(self.scaler, open('saved_models/scaler.pickle', 'wb'))
            pickle.dump(pca, open('saved_models/pca.pickle', 'wb'))
        columns = [f'pca_comp{x}' for x in range(1, N+1)]

        principalDf_train = pd.DataFrame(data = principalComponents_train
                    , columns = columns)

        x_val = self.scaler.transform(x_val)
        principalComponents_val = pca.transform(x_val)
        principalDf_val = pd.DataFrame(data = principalComponents_val
                    , columns = columns)
        

        # finalDf_train = pd.concat([principalDf_train, df_train[['Fazekas']]], axis = 1)
        # finalDf_val = pd.concat([principalDf_val, df_val[['Fazekas']]], axis = 1)

        self.datasets = {}
        self.datasets['train'] = {}
        self.datasets['train']['features'] = principalDf_train.to_dict(orient='list')
        self.datasets['val'] = {}
        self.datasets['val']['features'] = principalDf_val.to_dict(orient='list')
        self.datasets['train']['labels'] = df_fazekas_train
        self.datasets['val']['labels'] = df_fazekas_val



#%%


# for i in range(1, 6):
# print('ensamble: ', i)
classifer = MultiVariateClassifier(classification_type = 'tree', mode = 'custom', use_ensamble=True, n_estimators = 75, learning_rate = 1.97)
# classifer = MultiVariateClassifier(classification_type = 'tree', mode = 'None', max_depth = 2, use_ensamble=True, n_estimators= 7, learning_rate = 1)
classifer.add_features(N = 4) # Best N = 3
# classifer.add_dataset(data_train, set_type = 'train')
# classifer.add_dataset(data_val, set_type = 'val')
# classifer.add_features(feature_type = 'pca3')
# classifer.add_features(feature_type = 'entropy')
# classifer.add_features(feature_type = 'size')
# classifer.add_features(feature_type = 'lesions')
# classifer.test_features_normality()
# classifer.train_model_hyperparams()
# classifer.make_model()
# classifer.evaluate(verbatim=True, loadmodel = 'saved_models/best_adaboost.pickle', loadscaler = 'saved_models/scaler.pickle')
classifer.evaluate(verbatim=True, loadmodel = 'saved_models/best_adaboost.pickle')
# %%
# classifer.evaluate()
# classifer.roc_analysis()
# %%
