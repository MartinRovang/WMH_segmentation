# -*- coding: utf-8 -*-

import difflib
import numpy as np
import os
import SimpleITK as sitk
import scipy.spatial
import glob
import re
from rich.progress import track
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from skimage.morphology import remove_small_objects
# Set the path to the source data (e.g. the training data for self-testing)
# and the output directory of that subject

# C:\Users\Gimpe\Documents\GitHub\WMH-Segmentation\WMHSEG_0.2\dataanalysis\scannerID_2.csv
def compare_scanner_data(subjectname, path_txt = "/mnt/CRAI-NAS/all/martinsr/test_monai/WMHSEG_0.5/dataanalysis/scannerID_2.csv"):#path_txt:str = "/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/scannerID_2.csv"):
    data = pd.read_csv(path_txt, header = 0, error_bad_lines=False, delimiter = '\t',encoding = "ISO-8859-1")
    remove_data = ['subject-ass' ,'RepetitionTime', 'DeviceSerialNumber', 'ManufacturerModelName', 'SoftwareVersions', 'MagneticFieldStrength', 'RepetitionTime', 'EchoTime', 'FlipAngle']

    # print(np.array(data['subject-ass'].tolist()))
    file = np.where(data['subject-ass'].values == str(subjectname))
    extra_list = ['D110143PK-3', 'DD1233-201-1', 'D110091P-3', 'D110152-201-1', 'D110108P-3', 'D110101P-3', 'D110188-101-1', 'D110060P-3']
    # datainfo = data.iloc[file[0][0]]
    # print(datainfo)
    # get location of pandas dataframe with string
    if subjectname[0] == 'A':
        return 'Brno'# return the hospital and scanner corresponding to the given subject
    else:
        if len(file[0]) > 0:
            scannerinfo = data['ManufacturerModelName'].values[file[0]][0]
            return scannerinfo
        else:
            with open('metrics_analysis/scanner_nans.txt', 'a') as file:
                file.write(subjectname + '\n')
            
            if subjectname == 'D110015P-2':
                return 'Achieva'
            else:
                if subjectname in extra_list:
                    return 'Ingenia'
                else:
                    return 'nan'


def z_normalize(x):
    x = (x - np.mean(x))/np.std(x)
    return x


def do(path):
    """Main function"""
    # resultFilename = getResultFilename(participantDir)
    paths = glob.glob(path+'/*')
    patients_path_array = []

    print(paths)

    prediction_folder = path.split('/')[-1]
    
    to_output = f'metrics_analysis/2-5D_{prediction_folder}_test_edit.txt'
    all_metrics = f"""Subject, Dice, HD95, AVD, Lesion Detection, Lesion F1\n"""
    with open(to_output, 'a') as f:
        f.write(all_metrics)
    
    # Only find patients
    for path_ in paths:
        patient_num = re.findall(r'\d+', path_)
        if len(patient_num) > 0:
            patients_path_array.append(path_)

    avg_dsc = []    
    avg_h95 = []
    avg_avd = []
    avg_recall = []
    avg_f1 = []
    metrics = { 'dsc': [],
                'h95': [],
                'avd': [],
                'recall': [],
                'f1': [],
                'voxelsize': [],
                'scanner': []}
    for patient_path in track(patients_path_array, description="Iterating patients:"):

        # Grab patient folder path name
        l = str(patient_path)
        l = l.replace('\\', '/')
        l = l.split('/')
        patient_num = l[-1]
        # pred_path = patient_path+f'/{patient_num}_T1acq_nu_wmh_pred.nii.gz'
        # pred_path = patient_path+f'/WMH_{patient_num}.nii.gz'
        pred_path = patient_path+f'/F_seg_testseg.nii.gz'

        if not os.path.exists(pred_path):
            print(f'File does not exist {pred_path}, continuing...')
            continue

        annot_path = patient_path+'/reduced_an.nii.gz'
        FLAIR_path = patient_path+'/FLAIR.nii.gz'
        # resultImage = (nib.load(pred_path).get_fdata())# Mask loads in as 0-1
        # testImage = (nib.load(annot_path).get_fdata()) # Mask loads in as 0-1
        print(pred_path)
        print(annot_path)
        testImage, resultImage = getImages(annot_path, pred_path)

        try:
            FLAIRFILE = nib.load(FLAIR_path)
            header_info = FLAIRFILE.header
            sx, sy, sz = header_info.get_zooms()
            voxel_size = sx*sy*sz
        except Exception as e:
            print(e)

        # one, two, = sitk.GetArrayFromImage(testImage), sitk.GetArrayFromImage(resultImage)

        # for i in range(0, one.shape[0]):
        #     fig, ax = plt.subplots(1, 2)
        #     ax[0].imshow(one[i, :, :], cmap='gray')
        #     ax[1].imshow(two[i, :, :], cmap='gray')
        #     plt.show()

        if testImage and resultImage:
            # testImage[testImage > 0.89] = 1
            # testImage[testImage <= 0.89] = 0
            # testImage = testImage.astype('int')
            # resultImage[resultImage >= 0.5] = 1
            # resultImage[resultImage < 0.5] = 0
            # resultImage = resultImage.astype('int')
            
            dsc = getDSC(testImage, resultImage)
            h95 = getHausdorff(testImage, resultImage)
            avd = getAVD(testImage, resultImage)    
            recall, f1 = getLesionDetection(testImage, resultImage) 

            metrics['dsc'].append(dsc)
            metrics['h95'].append(h95)
            metrics['avd'].append(avd)
            metrics['voxelsize'].append(voxel_size)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            scannerdata = compare_scanner_data(patient_num)
            metrics['scanner'].append(scannerdata)
            # metrics['hospital'].append(hospitaldata)


            avg_dsc.append(dsc)
            avg_h95.append(h95)
            avg_avd.append(avd)
            avg_recall.append(recall)
            avg_f1.append(f1)

            print('Dice',                dsc,       '(higher is better, max=1)')
            print('HD',                  h95, 'mm',  '(lower is better, min=0)')
            print('AVD',                 avd,  '%',  '(lower is better, min=0)')
            print('Lesion detection', recall,       '(higher is better, max=1)')
            print('Lesion F1',            f1,       '(higher is better, max=1)')


            all_metrics = f"""{patient_num}, {dsc}, {h95}, {avd}, {recall}, {f1}\n"""


            # with open(patient_path+'/subject_id_metrics_NNUNET.txt', 'a') as f:
            # with open(patient_path+'/subject_id_metrics_crai.txt', 'a') as f:
            with open(to_output, 'a') as f:
                f.write(all_metrics)
            
            

            

        

    # Plot scores for different scanners
    row = 0
    col = 0
    fig, axes = plt.subplots(3, 2, figsize = (20, 15))
    for key in metrics:
        if key != 'scanner':
            b = sns.boxplot(x = metrics['scanner'], y = metrics[key], palette="rocket", ax=axes[row, col])
            axes[row, col].set_ylabel(key, fontsize = 20)
            b.tick_params(labelsize=15)
            col += 1
            if (col % 2) == 0:
                row += 1
                col = 0
    # b.axes.set_title("Title",fontsize=50)
    plt.savefig('metrics_analysis/eval_boxplot_val_edit.pdf')
    plt.close()

    # Plot scores for different scanners
    
    result = f"""
    AVG Dice,                {np.mean(avg_dsc)} +- {np.std(avg_dsc)} \n
    AVG HD,                  {np.mean(avg_h95)} +- {np.std(avg_h95)}\n
    AVG AVD,                 {np.mean(avg_avd)} +- {np.std(avg_avd)}\n 
    AVG Lesion detection,    {np.mean(avg_recall)} +- {np.std(avg_recall)}\n
    AVG Lesion F1',          {np.mean(avg_f1)} +- {np.std(avg_f1)} \n
    """

    print('AVG Dice',                f'{np.mean(avg_dsc)} +- {np.std(avg_dsc)}',       '(higher is better, max=1)')
    print('AVG HD',                  f'{np.mean(avg_h95)} +- {np.std(avg_h95)}', 'mm',  '(lower is better, min=0)')
    print('AVG AVD',                 f'{np.mean(avg_avd)} +- {np.std(avg_avd)}',  '%',  '(lower is better, min=0)')
    print('AVG Lesion detection', f'{np.mean(avg_recall)} +- {np.std(avg_recall)}',       '(higher is better, max=1)')
    print('AVG Lesion F1',            f'{np.mean(avg_f1)} +- {np.std(avg_f1)}',       '(higher is better, max=1)')

    with open(path+'/scoring.txt', 'w') as f:
        f.write(result)
    
    
    avg_dsc = z_normalize(avg_dsc)
    avg_h95 = z_normalize(avg_h95)
    avg_avd = z_normalize(avg_avd)
    avg_recall = z_normalize(avg_recall)
    avg_f1 = z_normalize(avg_f1)

    data_metrics = {'dsc': avg_dsc, 'h95': avg_h95, 'avd': avg_avd, 'recall': avg_recall, 'f1': avg_f1}
    lesionstats = pd.DataFrame(data_metrics)
    lesionstats.boxplot()
    plt.ylabel('z-standard score')
    plt.savefig('dataanalysis/boxplot_stats_metrics.pdf')
    plt.close()

    

def getImages(testFilename, resultFilename):
    """Return the test and result images, thresholded and non-WMH masked."""
    try:
        testImage   = sitk.ReadImage(testFilename)
        resultImage = sitk.ReadImage(resultFilename)

        # one, two, = sitk.GetArrayFromImage(testImage), sitk.GetArrayFromImage(resultImage)
        # one[one > 0.5] = 1
        # two[two > 0.5] = 1
        # one = one.astype('int').astype('bool')
        # two = two.astype('int').astype('bool')
        # one = remove_small_objects(one, min_size = 5, connectivity = 2, in_place = True)
        # one = remove_small_objects(one, min_size = 5, connectivity = 2, in_place = True)
        # two = remove_small_objects(two, min_size = 5, connectivity = 2, in_place = True)
        # two = remove_small_objects(two, min_size = 5, connectivity = 2, in_place = True)
        # testImage = sitk.GetImageFromArray(one)
        # resultImage = sitk.GetImageFromArray(two)
        
        # sitk.GetImageFromArray(a)  
        # one[one > 0.89] = 1
        # for i in range(0, one.shape[0]):
        #     if np.sum(one[i, :, :]) > 0 or np.sum(two[i, :, :]) > 0:
        #         fig, ax = plt.subplots(1, 2)
        #         ax[0].imshow(one[i, :, :], cmap='gray')
        #         ax[1].imshow(two[i, :, :], cmap='gray')
        #         plt.show()

        
        # Check for equality
        assert testImage.GetSize() == resultImage.GetSize()
        
        # Get meta data from the test-image, needed for some sitk methods that check this
        resultImage.CopyInformation(testImage)
        

        maskedTestImage = sitk.BinaryThreshold(testImage, 0.5,  999999999, 1, 0) # WMH == 1
        #resultImage = sitk.BinaryThreshold(resultImage, 5,  999999999, 1, 0) # WMH == 1       
 
        
        # #Convert to binary mask
        # if 'integer' in maskedResultImage.GetPixelIDTypeAsString():
        #     bResultImage = sitk.BinaryThreshold(maskedResultImage, 1, 100000, 1, 0)
        # else:
        #     bResultImage = sitk.BinaryThreshold(maskedResultImage, 0.5, 100000, 1, 0)
        resultImage = sitk.Cast(resultImage, sitk.sitkInt16)
        return maskedTestImage, resultImage
    except:
        return None, None
    

def getResultFilename(participantDir):
    """Find the filename of the result image.
    
    This should be result.nii.gz or result.nii. If these files are not present,
    it tries to find the closest filename."""
    files = os.listdir(participantDir)
    
    if not files:
        raise Exception("No results in "+ participantDir)
    
    resultFilename = None
    if 'result.nii.gz' in files:
        resultFilename = os.path.join(participantDir, 'result.nii.gz')
    elif 'result.nii' in files:
        resultFilename = os.path.join(participantDir, 'result.nii')
    else:
        # Find the filename that is closest to 'result.nii.gz'
        maxRatio = -1
        for f in files:
            currentRatio = difflib.SequenceMatcher(a = f, b = 'result.nii.gz').ratio()
            
            if currentRatio > maxRatio:
                resultFilename = os.path.join(participantDir, f)
                maxRatio = currentRatio
                
    return resultFilename
    
    
def getDSC(testImage, resultImage):    
    """Compute the Dice Similarity Coefficient."""
    testArray   = sitk.GetArrayFromImage(testImage).flatten()
    resultArray = sitk.GetArrayFromImage(resultImage).flatten()
    
    # similarity = 1.0 - dissimilarity
    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray) 
    

def getHausdorff(testImage, resultImage):
    """Compute the Hausdorff distance."""
    
    # Hausdorff distance is only defined when something is detected
    resultStatistics = sitk.StatisticsImageFilter()
    resultStatistics.Execute(resultImage)
    if resultStatistics.GetSum() == 0:
        return float('nan')
        
    # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
    eTestImage   = sitk.BinaryErode(testImage, (1,1,0) )
    eResultImage = sitk.BinaryErode(resultImage, (1,1,0) )
    
    hTestImage   = sitk.Subtract(testImage, eTestImage)
    hResultImage = sitk.Subtract(resultImage, eResultImage)    
    
    hTestArray   = sitk.GetArrayFromImage(hTestImage)
    hResultArray = sitk.GetArrayFromImage(hResultImage)   
        
    # Convert voxel location to world coordinates. Use the coordinate system of the test image
    # np.nonzero   = elements of the boundary in numpy order (zyx)
    # np.flipud    = elements in xyz order
    # np.transpose = create tuples (x,y,z)
    # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
    testCoordinates   = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hTestArray) ))]
    resultCoordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hResultArray) ))]
        
            
    # Use a kd-tree for fast spatial search
    def getDistancesFromAtoB(a, b):    
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]
    
    # Compute distances from test to result; and result to test
    dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
    dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)    
    
    return max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))
    
    
def getLesionDetection(testImage, resultImage):    
    """Lesion detection metrics, both recall and F1."""
    
    # Connected components will give the background label 0, so subtract 1 from all results
    ccFilter = sitk.ConnectedComponentImageFilter()    
    ccFilter.SetFullyConnected(True)
    
    # Connected components on the test image, to determine the number of true WMH.
    # And to get the overlap between detected voxels and true WMH
    ccTest = ccFilter.Execute(testImage)    
    lResult = sitk.Multiply(ccTest, sitk.Cast(resultImage, sitk.sitkUInt32))
    
    ccTestArray = sitk.GetArrayFromImage(ccTest)
    lResultArray = sitk.GetArrayFromImage(lResult)
    
    # recall = (number of detected WMH) / (number of true WMH) 
    nWMH = len(np.unique(ccTestArray)) - 1
    if nWMH == 0:
        recall = 1.0
    else:
        recall = float(len(np.unique(lResultArray)) - 1) / nWMH
    
    # Connected components of results, to determine number of detected lesions
    ccResult = ccFilter.Execute(resultImage)
    lTest = sitk.Multiply(ccResult, sitk.Cast(testImage, sitk.sitkUInt32))
    
    ccResultArray = sitk.GetArrayFromImage(ccResult)
    lTestArray = sitk.GetArrayFromImage(lTest)
    
    # precision = (number of detections that intersect with WMH) / (number of all detections)
    nDetections = len(np.unique(ccResultArray)) - 1
    if nDetections == 0:
        precision = 1.0
    else:
        precision = float(len(np.unique(lTestArray)) - 1) / nDetections
    
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    
    return recall, f1    

    
def getAVD(testImage, resultImage):   
    """Volume statistics."""
    # Compute statistics of both images
    testStatistics   = sitk.StatisticsImageFilter()
    resultStatistics = sitk.StatisticsImageFilter()
    
    testStatistics.Execute(testImage)
    resultStatistics.Execute(resultImage)
        
    return float(abs(testStatistics.GetSum() - resultStatistics.GetSum())) / float(testStatistics.GetSum()) * 100
    
if __name__ == "__main__":
    # do(path = r"C:\Users\Gimpe\Google Drive\Master -Signal_processingWORK\Masteroppgave\Main_code\data\val")    
    do(path = "/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/alldata")    
    # do(path = "/mnt/HDD18TB/martinsr/alldata2")    
