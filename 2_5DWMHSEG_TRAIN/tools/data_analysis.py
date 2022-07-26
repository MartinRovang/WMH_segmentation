from re import S
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import pandas as pd
from scipy.stats import entropy
from scipy.stats import shapiro
from skimage.measure import label, regionprops
import pickle
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

np.random.seed(1)




def compare_scanner_data(subjectname, path_txt = r"C:\Users\MartinR\Documents\GitHub\WMH-Segmentation\WMHSEG_0.2\dataanalysis\scannerID_2.csv"):#path_txt:str = "/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/scannerID_2.csv"):
    data = pd.read_csv(path_txt, header = 0, error_bad_lines=False, delimiter = '\t',encoding = "ISO-8859-1")
    remove_data = ['subject-ass' ,'RepetitionTime', 'DeviceSerialNumber', 'ManufacturerModelName', 'SoftwareVersions', 'MagneticFieldStrength', 'RepetitionTime', 'EchoTime', 'FlipAngle']

    print(np.array(data['subject-ass'].tolist()))
    file = np.where(data['subject-ass'].values == str(subjectname))
    # datainfo = data.iloc[file[0][0]]
    # print(datainfo)
    # get location of pandas dataframe with string
    if subjectname[0] == 'A':
        return 'Brno'# return the hospital and scanner corresponding to the given subject
    else:
        if len(file[0]) > 0:
            scannerinfo = data['Manufacturer'].values[file[0]][0]
            return scannerinfo
        else:
            return 'nan'

class AnalysisModule():
    def __init__(self, mode = 'custom'):
        self.mode = mode
        self.datasets = {'train': {'features':{}}, 'val': {'features':{}}}
        self.datasets_rad = {}
        if mode == 'custom':
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
    

    def add_features(self, feature_type:str = 'entropy', N = 4):
        if self.mode == 'custom':
            self.analyze_best_features(N)


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

    def boxplot(self, x:str='labels', datasettype:str='train'):
        """Plots the boxplots for all the features in brackets of the labels"""
        N_features_rows = len(self.datasets[datasettype]['features'])
        fig, axes = plt.subplots(int(np.ceil(N_features_rows/3)), 3, figsize=(15,10))
        
        row = 0
        col = 0
        for feature_type in self.datasets[datasettype]['features']:
            data = {'labels': self.datasets[datasettype]['features'][x], feature_type: self.datasets[datasettype]['features'][feature_type]}
            sns.boxplot(data = data, x = 'labels', y = feature_type, palette="rocket", ax=axes[row, col])
            axes[row, col].set_xlabel('Labels')
            axes[row, col].set_ylabel(feature_type)
            col += 1
            if (col % 3) == 0:
                row += 1
                col = 0
    
        plt.savefig('dataanalysis/boxplot.png')
        plt.tight_layout()
        plt.show()
        plt.close()
    
    def correlation(self):
        """plots the correlation between the features"""
        if self.mode == 'custom':
            columns=[key for key in self.datasets['train']['features']]
            df = pd.DataFrame(self.datasets['train']['features'], columns=columns)
            corrMatrix = df.corr(method = 'pearson')
            heat = sns.heatmap(corrMatrix, annot=True)
            heat.set_title('Pearson correlation')
            plt.savefig('dataanalysis/heat.png')
            plt.show()
            plt.close()

        else:
            data = {}
            for feature in self.datasets['train']['features']:
                feat = self.datasets['train']['features'][feature]
                data[feature] = feat
            
            data['Labels'] = train_label
            columns=[key for key in data]
            df = pd.DataFrame(data, columns=columns)

            corrMatrix = df.corr(method = 'pearson')
            heat = sns.heatmap(corrMatrix, annot=True)
            heat.set_title('Pearson correlation')
            plt.savefig('dataanalysis/heat.png')
            plt.close()
    
    def analyze_best_features(self, N:int = 4):
        """
        Performes PCA on the radiometrics data to find the features which describes the variance the most. 
        Materials:
        #https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
        #https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
        """
        
        remove_list = ['diagnostics_Configuration_Settings', 'diagnostics_Versions_PyRadiomics', 'diagnostics_Versions_Numpy', 'diagnostics_Versions_SimpleITK', 'diagnostics_Versions_PyWavelet', 'diagnostics_Versions_Python', 'diagnostics_Configuration_EnabledImageTypes', 'diagnostics_Image-original_Hash', 'diagnostics_Image-original_Dimensionality', 'diagnostics_Mask-original_Hash', 'diagnostics_Image-original_Mean', 'diagnostics_Mask-original_VoxelNum', 'diagnostics_Mask-original_VolumeNum', 'Image', 'diagnostics_Image-original_Minimum', 'diagnostics_Image-original_Maximum', 'diagnostics_Image-original_Spacing', 'diagnostics_Image-original_Size', 'diagnostics_Mask-original_Size', 'diagnostics_Mask-original_Spacing', 'diagnostics_Mask-original_BoundingBox', 'diagnostics_Versions_PyRadiomics', 'diagnostics_Versions_Numpy']
        columns=[key for key in self.datasets['train']['features'] if key not in remove_list and type(self.datasets['train']['features'][key][0]) != tuple]
        df_train = pd.DataFrame(self.datasets['train']['features'], columns=columns)
        df_val = pd.DataFrame(self.datasets['val']['features'], columns=columns)

        df_fazekas_train = df_train[['Fazekas']]
        df_fazekas_val = df_val[['Fazekas']]
        df_train = df_train.drop('Fazekas', axis = 1)
        df_val = df_val.drop('Fazekas', axis = 1)


        # Separating out the features
        x_train = df_train.loc[:, df_train.columns].values
        x_val = df_val.loc[:, df_val.columns].values

        # Standardizing the features
        x_train = StandardScaler().fit_transform(x_train)
        x_val = StandardScaler().fit_transform(x_val)

        pca = PCA(n_components=N).fit(x_train)
        columns = [f'pca_comp{x}' for x in range(1, N+1)]
        principalComponents_train = pca.transform(x_train)
        principalDf_train = pd.DataFrame(data = principalComponents_train
                    , columns = columns)
        principalComponents_val = pca.transform(x_val)
        principalDf_val = pd.DataFrame(data = principalComponents_val
                    , columns = columns)
        

        finalDf_train = pd.concat([principalDf_train, df_fazekas_train], axis = 1)
        finalDf_val = pd.concat([principalDf_val, df_fazekas_val], axis = 1)

        self.datasets = {}
        self.datasets['train'] = {}
        self.datasets['train']['features'] = finalDf_train.to_dict(orient='list')
        self.datasets['val'] = {}
        self.datasets['val']['features'] = finalDf_val.to_dict(orient='list')


# train_load = np.load('train.npz')
# val_load = np.load('val.npz')

# X_train, train_label, voxel_train = train_load['data'], train_load['labels'], train_load['voxel']
# X_val, val_label, voxel_val = val_load['data'], val_load['labels'], val_load['voxel']

# X_train = X_train#*voxel_train[:, None]
# X_val = X_val#*voxel_val[:, None]


# data_train = {'data': X_train, 'labels': train_label}
# data_val = {'data': X_val, 'labels': val_label}



anal = AnalysisModule(mode = 'custom')
anal.add_features(N = 7)
anal.boxplot(x = 'Fazekas')
anal.correlation()


# anal.add_dataset(data_train, set_type = 'train')
# anal.add_dataset(data_val, set_type = 'val')
# anal.add_features(feature_type = 'entropy')
# anal.add_features(N = 7)
# anal.add_features(feature_type = 'size')
# anal.add_features(feature_type = 'pca3')
# anal.add_features(feature_type = 'lesions')
# anal.test_features_normality()
# anal.boxplot(x = 'Fazekas')
# anal.correlation()
