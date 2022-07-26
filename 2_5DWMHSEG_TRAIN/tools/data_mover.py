import shutil 
import glob


def get_test_data(path_to_experiment):
    with open(f'{path_to_experiment}/testdatasplit.txt', 'r') as f:
        datadic = eval(f.read())
    return datadic


def copy_files(files_list,  path_to_copy):
    for file in files_list:
        try:
            file = str(file['image'])
            file = "/".join(file.split('/')[:-1])
            output = path_to_copy+'/'+file.split('/')[-1]
            print(file, output)
            shutil.copytree(file, output)
        except Exception as e:
            print(e)
    print('Copied files to: ', path_to_copy)


def copy_T1_files(files_list,  path_to_copy):
    for file in files_list:
        try:
            file = str(file['image'])
            file = "/".join(file.split('/')[:-1])
            id = file.split('/')[-1]
            file = f'/mnt/HDD16TB/martinsr/DatasetWMH211018/Dataset_WMH_211018/{id}_T1_BFCorr.nii.gz'
            output = path_to_copy+'/'+id+f'/{id}_T1_BFCorr.nii.gz'
            print(file, output)
            shutil.copy(file, output)
        except Exception as e:
            print(e)
    print('Copied files to: ', path_to_copy)


testdata = get_test_data('/mnt/CRAI-NAS/all/martinsr/test_monai/WMHSEG_0.5/outputs/2022-04-27/12-23-20')


path_to_copy = '/mnt/CRAI-NAS/all/martinsr/test_monai/DatasetWMH211018_v2_testdata'
copy_T1_files(testdata, path_to_copy)