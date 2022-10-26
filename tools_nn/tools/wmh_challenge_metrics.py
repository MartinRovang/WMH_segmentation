# -*- coding: utf-8 -*-

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
import datetime



class WmhChallengeStats:
    def __init__(self):
        self.dsc_participant = {}
        self.h95_participant = {}
        self.avd_participant = {}
        self.recall_participant = {}
        self.f1_participant = {}


    def compute_stats(self, annot_path, pred_path, model_name):
        
        # check if paths exists
        if not os.path.exists(annot_path):
            raise Exception(f"No annotation path found, {pred_path}")
        if not os.path.exists(pred_path):
            raise Exception(f"No prediction path found, {pred_path}")    
    
        testImage, resultImage = self.getImages(annot_path, pred_path)

        if testImage and resultImage:
            dsc = self.getDSC(testImage, resultImage)
            h95 = self.getHausdorff(testImage, resultImage)
            avd = self.getAVD(testImage, resultImage)    
            recall, f1 = self.getLesionDetection(testImage, resultImage)

            # print
            print(pred_path)
            print("DSC: ", dsc)
            print("H95: ", h95)
            print("AVD: ", avd)
            print("Recall: ", recall)
            print("F1: ", f1)

            try:
                self.dsc_participant[os.path.basename(pred_path).replace(".nii.gz", "")][model_name] = dsc
                self.h95_participant[os.path.basename(pred_path).replace(".nii.gz", "")][model_name] = h95
                self.avd_participant[os.path.basename(pred_path).replace(".nii.gz", "")][model_name] = avd
                self.recall_participant[os.path.basename(pred_path).replace(".nii.gz", "")][model_name] = recall
                self.f1_participant[os.path.basename(pred_path).replace(".nii.gz", "")][model_name] = f1
            except:
                self.dsc_participant[os.path.basename(pred_path).replace(".nii.gz", "")] = {model_name: dsc}
                self.h95_participant[os.path.basename(pred_path).replace(".nii.gz", "")] = {model_name: h95}
                self.avd_participant[os.path.basename(pred_path).replace(".nii.gz", "")] = {model_name: avd}
                self.recall_participant[os.path.basename(pred_path).replace(".nii.gz", "")] = {model_name: recall}
                self.f1_participant[os.path.basename(pred_path).replace(".nii.gz", "")] = {model_name: f1}
    
    def write_stats(self, extra):
        """Write the stats to a csv file."""
        # Create the dataframe
        df = pd.DataFrame(columns=["participant", "model", "dsc", "h95", "avd", "recall", "f1"])

        # Iterate over all participants
        for participant in self.dsc_participant.keys():
            # Iterate over all models
            for model in self.dsc_participant[participant].keys():
                # Add the row
                df = df.append({"participant": participant, "model": model, "dsc": self.dsc_participant[participant][model], "h95": self.h95_participant[participant][model], "avd": self.avd_participant[participant][model], "recall": self.recall_participant[participant][model], "f1": self.f1_participant[participant][model]}, ignore_index=True)
        
        # sort by participant id
        df = df.sort_values(by=["participant"])
        df.to_csv("./../report/stats_{}_{}.csv".format(datetime.datetime.now().strftime("%Y-%m-%d"), extra), index=False)
    
    def compute_folder_stats(self, annot_folder, pred_folder, model_name):
        """Compute the stats for all participants in the given folder."""
        # Get all participants
        participants = os.listdir(annot_folder)

        # Iterate over all participants
        for participant in track(participants):
            # Get the participant directory
            participant_path_pred = os.path.join(pred_folder, participant)
            participant_path_annot = os.path.join(annot_folder, participant)

            # Compute the stats
            self.compute_stats(participant_path_annot, participant_path_pred, model_name)

    def getImages(self, testFilename, resultFilename):
        """Return the test and result images, thresholded and non-WMH masked."""
        try:
            testImage   = sitk.ReadImage(testFilename)
            resultImage = sitk.ReadImage(resultFilename)


            # Check for equality
            assert testImage.GetSize() == resultImage.GetSize()
            
            resultImage.CopyInformation(testImage)

            # set all image voxels to from 0 to 1 to 1
            # resultImage = sitk.BinaryThreshold(resultImage, lowerThreshold=0, upperThreshold=1.5, insideValue=1, outsideValue=0)
            # testImage   = sitk.BinaryThreshold(testImage, lowerThreshold=0, upperThreshold=1.5, insideValue=1, outsideValue=0)

            resultImage = sitk.Cast(resultImage, sitk.sitkInt16)
            testImage = sitk.Cast(testImage, sitk.sitkInt16)

            testArray   = sitk.GetArrayFromImage(testImage).flatten()
            resultArray = sitk.GetArrayFromImage(resultImage).flatten()
            assert np.all(np.isin(testArray, [0, 1])), " ground truth contains values other than 0 and 1 {}".format(np.unique(testArray))
            assert np.all(np.isin(resultArray, [0, 1])),"predict image contains values other than 0 and 1 {}".format(np.unique(resultArray))

            return testImage, resultImage
        except Exception as e:
            print(e)
            exit()
        
    
    def getDSC(self, testImage, resultImage):    
        """Compute the Dice Similarity Coefficient."""
        testArray   = sitk.GetArrayFromImage(testImage).flatten()
        resultArray = sitk.GetArrayFromImage(resultImage).flatten()
        
        # similarity = 1.0 - dissimilarity
        return 1.0 - scipy.spatial.distance.dice(testArray, resultArray) 
    

    def getHausdorff(self, testImage, resultImage):
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
    
    
    def getLesionDetection(self, testImage, resultImage):    
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

    
    def getAVD(self, testImage, resultImage):   
        """Volume statistics."""
        # Compute statistics of both images
        testStatistics   = sitk.StatisticsImageFilter()
        resultStatistics = sitk.StatisticsImageFilter()
        
        testStatistics.Execute(testImage)
        resultStatistics.Execute(resultImage)
            
        return float(abs(testStatistics.GetSum() - resultStatistics.GetSum())) / float(testStatistics.GetSum()) * 100

model_name = "25DUNet"
# pred_folder ="/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/predictions_epoch=550-dice_mean=77_44_task=15_fold=0_tta"
# pred_folder = "/mnt/CRAI-NAS/all/martinsr/test_monai/DatasetWMH211018_v2_newphase/pred"
pred_folder = "/mnt/CRAI-NAS/all/martinsr/test_monai/WMH-Segmentation_Production/console_version/result_test"
# annot_folder = "/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/labelsBrno"
annot_folder = "/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/labelsTs"

# annot_folder = "/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/labelsBrno"
# pred_folder = "/mnt/CRAI-NAS/all/martinsr/test_monai/test_brno_sample_newphase/pred"
stats  = WmhChallengeStats()
stats.compute_folder_stats( annot_folder, pred_folder, model_name = model_name)
stats.write_stats(f"test_{model_name}")


dsc = []
hausdorff = []
recall = []
f1 = []
avd = []

for key in stats.dsc_participant.keys():
    dsc.append(stats.dsc_participant[key][model_name])
    hausdorff.append(stats.h95_participant[key][model_name])
    recall.append(stats.recall_participant[key][model_name])
    f1.append(stats.f1_participant[key][model_name])
    avd.append(stats.avd_participant[key][model_name])
    
# mean +- std

dsc_mean = np.nanmean(dsc)
dsc_std = np.nanstd(dsc)
hausdorff_mean = np.nanmean(hausdorff)
hausdorff_std = np.nanstd(hausdorff)
recall_mean = np.nanmean(recall)
recall_std = np.nanstd(recall)
f1_mean = np.nanmean(f1)
f1_std = np.nanstd(f1)
avd_mean = np.nanmean(avd)
avd_std = np.nanstd(avd)


print("DSC: ", dsc_mean, "+-", dsc_std)
print("Hausdorff: ", hausdorff_mean, "+-", hausdorff_std)
print("Recall: ", recall_mean, "+-", recall_std)
print("F1: ", f1_mean, "+-", f1_std)
print("AVD: ", avd_mean, "+-", avd_std)

with open(f"./../report/average_{model_name}_test.txt", "w") as f:
    f.write("DSC: "+str(dsc_mean)+"+-"+str(dsc_std)+"\n")
    f.write("Hausdorff: "+str(hausdorff_mean)+"+-"+str(hausdorff_std)+"\n")
    f.write("Recall: "+str(recall_mean)+"+-"+str(recall_std)+"\n")
    f.write("F1: "+str(f1_mean)+"+-"+str(f1_std)+"\n")
    f.write("AVD: "+str(avd_mean)+"+-"+str(avd_std)+"\n")