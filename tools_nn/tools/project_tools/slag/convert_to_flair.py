# # Some steps using FSL tools
# # used 7 degrees of freedom and mutual information cost function
# python 2.7
import subprocess
import glob
path = "./imagesTr_FLAIR/*.nii.gz"


for flairfile in glob.glob(path):
    #bet sub-100343_ses-study-MR_3D_FLAIR.nii.gz bet_FLAIR
    id_ = flairfile.split("/")[-1].split(".")[0]
    dwi_mask_path = flairfile.replace("imagesTr_FLAIR", "labelsTr")
    outpath = "./labelsTr_FLAIR/{}".format(id_)
    dwi_path = flairfile.replace("imagesTr_FLAIR", "imagesTr")
    subprocess.call(["bet", flairfile, "bet_FLAIR"])
    #bet sub-100343_ses-study-MR_DWI.nii.gz bet_DWI
    subprocess.call(["bet", dwi_path, "bet_DWI"])
    #flirt -in ../func/bet_DWI.nii.gz -ref bet_FLAIR.nii.gz -out bet_DWI_to_bet_FLAIR -omat mat_bet_DWI_to_bet_FLAIR.mat -cost mutualinfo -dof 7 
    subprocess.call(["flirt", "-in", "bet_DWI.nii.gz", "-ref", "bet_FLAIR.nii.gz", "-out", "bet_DWI_to_bet_FLAIR", "-omat", "mat_bet_DWI_to_bet_FLAIR.mat", "-cost", "mutualinfo", "-dof", "7"])
    #flirt -in mask/sub-100343.nii.gz -ref anat/bet_FLAIR.nii.gz -applyxfm -init anat/mat_bet_DWI_to_bet_FLAIR.mat -out maskDWI_to_FLAIR    
    subprocess.call(["flirt", "-in", dwi_mask_path, "-ref", "bet_FLAIR.nii.gz", "-applyxfm", "-init", "mat_bet_DWI_to_bet_FLAIR.mat", "-out", outpath])
    print "Done with {}".format(id_)




# bet sub-100343_ses-study-MR_3D_FLAIR.nii.gz bet_FLAIR
# bet sub-100343_ses-study-MR_DWI.nii.gz bet_DWI
# flirt -in ../func/bet_DWI.nii.gz -ref bet_FLAIR.nii.gz -out bet_DWI_to_bet_FLAIR -omat mat_bet_DWI_to_bet_FLAIR.mat -cost mutualinfo -dof 7 
# flirt -in mask/sub-100343.nii.gz -ref anat/bet_FLAIR.nii.gz -applyxfm -init anat/mat_bet_DWI_to_bet_FLAIR.mat -out maskDWI_to_FLAIR


