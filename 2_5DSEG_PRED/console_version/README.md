
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/CRAI-OUS/WMH-Segmentation_Production">
    <img src="logo_new.jpg" alt="Logo" width="766" height="494">
  </a>

  <h3 align="center">WMH segmentation production</h3>

  <p align="center">
    WMH segmentation using FLAIR and/or T1 on MR volumes
    <br />
    <a href="https://github.com/CRAI-OUS/WMH-Segmentation_Productionn"><strong>Explore the docs »</strong></a><br>
    <br />
    <br />
    <a href="https://github.com/CRAI-OUS/WMH-Segmentation_Production/issues">Report Bug</a>
    ·
    <a href="https://github.com/CRAI-OUS/WMH-Segmentation_Production/issues">Request Feature</a>
  </p>
</p>


### Run on your own computer

##### 1. Download repo

##### 2. Make sure you have all dependencies, pip install -r requirements.txt

##### 3. Change target directory in <code>config.py</code> `@ weights: "weights/wmh_model_unique_data.pth"`  and the model type : UNET or UUNET (UNET by default) 

##### 4. Make sure you have your data in the "data" folder in the same directory.

##### 5. Run <code>main_prediction.py</code>

#### Predictions will be made inside the subject folders with the name <code>F_seg.nii.gz</code>

```
DSC:  0.6808092008717108 +- 0.1749727617385947
Hausdorff:  9.751399615092273 +- 9.279947874717204
Recall:  0.7914909337086632 +- 0.16611485702407042
F1:  0.5269392661792544 +- 0.1548622287067465
AVD:  55.08561295745984 +- 54.99577215339091
```

### Data folder structure:
-data
  --ID1.nii.gz
  --ID2.nii.gz
  --ID3.nii.gz
  ...