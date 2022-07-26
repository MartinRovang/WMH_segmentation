
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

##### 3. Change target directory in <code>config.py</code> and the model type : UNET or UUNET (UNET by default)

##### 4. Run <code>main_prediction.py</code>

#### Predictions will be made inside the subject folders with the name <code>F_seg.nii.gz</code>

```
AVG Dice,                0.7385201129452749 

AVG HD,                  6.519434054410435 

AVG AVD,                 38.330993258895006 

AVG Lesion detection,    0.6300820370625283 

AVG Lesion F1',          0.5453687214850291 
```

Fazekas predictions are now added into the pipeline. Note: Fazekas has very few data points, hence, not be predicted with great accuracy.

The Fazekas prediction will be outputted to `fazekas_scale.txt`.

```
              precision    recall  f1-score   support

   fazekas 0       0.84      0.94      0.89        17
   fazekas 1       0.85      0.68      0.76        25
   fazekas 2       0.67      1.00      0.80        10
   fazekas 3       1.00      0.33      0.50         3

    accuracy                           0.80        55
   macro avg       0.84      0.74      0.74        55
weighted avg       0.82      0.80      0.79        55
```
