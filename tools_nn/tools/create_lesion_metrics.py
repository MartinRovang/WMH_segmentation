from itertools import count
import SimpleITK as sitk
import numpy as np
import glob
from rich.table import Table
from rich.console import Console
import pandas as pd

class LesionMetric:
    def __init__(self, path_gt:str, path_target:str):
        self.all_gt = glob.glob(path_gt+'/*.nii.gz')
        self.path_target = path_target
    

    def get_lesions(self, output_extra):
        # make rich table
        # table = Table(title="Lesion Metrics")
        # # "Average Lesion Size", "Total segmentation", "Total TP", "Total FN", "Total FP", "Average Recall"
        # table.add_column("ID", justify="left" , style="cyan")
        # table.add_column("Average Lesion Size", justify="left" , style="cyan")
        # table.add_column("Total segmentation", justify="left" , style="cyan")
        # table.add_column("Total TP", justify="left" , style="cyan")
        # table.add_column("Total FN", justify="left" , style="cyan")
        # table.add_column("Total FP", justify="left" , style="cyan")
        # table.add_column("Average Recall", justify="left" , style="cyan")
        table_pandas = {"ID": [], "Lesion Size": [], "Voxel TP": [], "Voxel FN": [], "Recall": []}
        table_pandas_fp = {"ID": [], "Lesion Size": [], "Voxel FP": [], "Voxel TP": [], "Precision": []}

        for gt in self.all_gt:
            id = gt.split('/')[-1]
            target = self.path_target + '/' + id
            gt = sitk.ReadImage(gt)
            target = sitk.ReadImage(target)
            spacing = gt.GetSpacing()
            target_array = sitk.GetArrayFromImage(target)
            gt_array = sitk.GetArrayFromImage(gt)
            # copy same metadata as gt
            # target.CopyInformation(gt)
            # print metadata
            # to int8
            gt = sitk.Cast(gt, sitk.sitkInt8)
            target = sitk.Cast(target, sitk.sitkInt8)
            # get individual lesions using connected components
            ccFilter = sitk.ConnectedComponentImageFilter()    
            ccFilter.SetFullyConnected(True)
            # Connected components on the test image, to determine the number of true WMH.
            # And to get the overlap between detected voxels and true WMH
            ccTest = ccFilter.Execute(gt)    
            cctarget = ccFilter.Execute(target)

            ccTestArray = sitk.GetArrayFromImage(ccTest)
            cctargetArray = sitk.GetArrayFromImage(cctarget)
            
            unique_lesions_gt, gt_lesion_count = np.unique(ccTestArray, return_counts=True)
            unique_lesions_target, target_lesion_count = np.unique(cctargetArray, return_counts=True)
            unique_lesions_gt = unique_lesions_gt[1:]
            unique_lesions_target = unique_lesions_target[1:]
            gt_lesion_count = gt_lesion_count[1:]
            target_lesion_count = target_lesion_count[1:]


            lesion_recall = []
            lesion_sizes = []
            for lesion, count in zip(unique_lesions_gt, gt_lesion_count):
                lesion_idx = np.where(ccTestArray == lesion)
                TP = (target_array[lesion_idx] == 1).sum()
                FN = (target_array[lesion_idx] == 0).sum()

                recall = TP / (TP + FN)
                lesion_recall.append(recall)
            
                lesion_size = count*0.001*spacing[0]*spacing[1]*spacing[2]
                lesion_sizes.append(lesion_size)

                table_pandas["Lesion Size"].append(lesion_size)
                table_pandas["Voxel TP"].append(TP)
                table_pandas["Voxel FN"].append(FN)
                table_pandas["Recall"].append(recall)
                table_pandas["ID"].append(id)

            for lesion, count in zip(unique_lesions_target, target_lesion_count):
                lesion_size_fp = count*0.001*spacing[0]*spacing[1]*spacing[2]
                table_pandas_fp["Lesion Size"].append(lesion_size_fp)
                lesion_idx = np.where(cctargetArray == lesion)
                FP = (gt_array[lesion_idx] == 0).sum()
                TP = (gt_array[lesion_idx] == 1).sum()

                precision = TP / (TP + FP)
                
                table_pandas_fp["Voxel FP"].append(FP)
                table_pandas_fp["Voxel TP"].append(TP)
                table_pandas_fp["Precision"].append(precision)
                table_pandas_fp["ID"].append(id)


            # table.add_row(id, str(round(np.mean(lesion_sizes), 3)), str(round(lesion_total, 3)), str(lesion_tp), str(lesion_fn), str(lesion_fp), str(round(np.mean(lesion_recall), 3)))
            # console = Console()
            # console.print(table)


        # make pandas table
        df = pd.DataFrame(table_pandas)
        df.to_csv(f'lesion_metrics_{output_extra}.csv', index=False)

        df_fp = pd.DataFrame(table_pandas_fp)
        df_fp.to_csv(f'lesion_metrics_fp_{output_extra}.csv', index=False)

        # get row of all lesions within Average Lesion Size of [0, 0.05]
        # df_lesion_size = df[df['Average Lesion Size'] < 0.05]




if __name__ == '__main__':
    path_gt = '/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/labelsTs'
    # path_target = '/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/predictions_epoch=550-dice_mean=77_44_task=15_fold=0_tta_test'
    # path_target = "/mnt/CRAI-NAS/all/martinsr/test_monai/DatasetWMH211018_v2_newphase/pred"
    path_target = "/mnt/CRAI-NAS/all/martinsr/test_monai/WMH-Segmentation_Production/console_version/result_test"
    lm = LesionMetric(path_gt, path_target)
    lm.get_lesions(output_extra = "25D_test")