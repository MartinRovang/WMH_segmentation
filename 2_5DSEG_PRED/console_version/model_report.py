#%%
import numpy as np
import glob
import nibabel as nib
from skimage.measure import label, regionprops
import subprocess
import os



main_path = "/mnt/CRAI-NAS/all/martinsr/test_monai/WMH-Segmentation_Production/console_version/"
subpaths = ['data']
for subpath in subpaths:
    path = main_path+subpath
    nifty_files = glob.glob(path+'/*')

    start = f"""
    \\documentclass[a4paper]{{article}}
    \\usepackage{{listings}}
    \\usepackage[english]{{babel}}
    \\usepackage[utf8x]{{inputenc}}
    \\usepackage[T1]{{fontenc}}
    \\usepackage{{listings}}
    \\usepackage{{float}}
    \\usepackage{{graphicx}}
    \\usepackage{{geometry}}
    \\geometry{{
    a4paper,
    top=20mm,
    }}
    \\title{{Segmentation report}}
    \\date{{\\today}}
    \\batchmode
    \\begin{{document}}
    \\maketitle
    """

    end = f"""\\end{{document}}"""

    table_begin = """
    \\begin{table}[]
    \\begin{tabular}{lll}
    ID & Total WMH [ml] & Amount of lesions \\\ \hline
    """

    table_end = f"""
    \\end{{tabular}}
    \\end{{table}}
    """

    all_tables = table_begin
    Errors = ''


    for nifty in nifty_files:
        try:
            nifty_file = nifty+'/F_seg.nii.gz'
            patient_id = nifty.split('/')[-1]
            sx, sy, sz = nib.load(nifty_file).header.get_zooms()
            vol = (sx * sy * sz)
            data = nib.load(nifty_file).get_fdata()
            mask_label_ = label(data, connectivity=2)
            mask_prop = regionprops(mask_label_)
            data_volume = np.sum(data)*vol*0.001
            amount_of_lesions = len(mask_prop)
            total_lesion_size = str(round(data_volume,2))
            amount_of_lesions = str(round(amount_of_lesions,2))

            all_tables += f"""
            {patient_id} &     {total_lesion_size}              &    {amount_of_lesions}               \\\ 
            """
        except Exception as e:
            print(e)
            print(nifty_file)
            Errors += f""" Could not load; {patient_id} \n"""
            continue

    all_tables += table_end

    document = start + all_tables + Errors + end

    with open(f'report/segmentation_summary_{subpath}.tex', 'w') as f:
        f.write(document)


    # '-interaction=batchmode'
    os.chdir('report')
    cmd = subprocess.Popen(['pdflatex', f'segmentation_summary_{subpath}.tex'])
    cmd.communicate()
    os.chdir('../')

# %%


    # \\begin{{figure}}[H]
    #     \includegraphics[width=0.90\\textwidth]{{learning_rate.pdf}}
    #     \caption{{Learning rate plot over epochs.}}
    #     \label{{learning_rate_plot}}
    # \\end{{figure}}