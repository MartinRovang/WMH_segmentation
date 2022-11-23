
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/CRAI-OUS/WMH-Segmentation_Production">
    <img src="https://github.com/CRAI-OUS/WMH-Segmentation_Experiment/raw/main/images/Logo_splatter.jpg" alt="Logo" width="498" height="460">
  </a>

  <p align="center">
    WMH segmentation using FLAIR.
    <br />
    <a href="https://arxiv.org/abs/2207.08467"><strong>Research article »</strong></a><br>
    <br />
    <br />
    <a href="https://github.com/MartinRovang/WMH_segmentation/issues">Report Bug</a>
    ·
    <a href="https://github.com/MartinRovang/WMH_segmentation/issues">Request Feature</a>
  </p>
</p>

Weights can be recived on request.

### Run on your own computer

#### 1. Clone repo.

#### 2. Change directory to repo.

#### 3. Add datafolder in current format;
```
-data
--imagesTr
---ID1.nii.gz
   ...
   
--labelsTr
---ID1.nii.gz
   ...

--imagesvalTr
---ID5.nii.gz
   ...
   
--labelsvalTr
---ID5.nii.gz
   ...

```
#### 4. Run command `docker build -t WMHSEG `. 

#### 5. Run command  `docker run -it --gpus all --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864  --rm -v ${PWD}/:/workspace wmh:latest /bin/bash`.

#### 6. Run command  `cd /workspace`.

# For training (Needs scanner datafile)

#### 1. Run command  `cd /2_5DWMHSEG_TRAIN`.
#### 2. Change the conf/config.yaml file.
#### 3. Run command  `python main.py`.

# For prediction (Needs weight file)
#### 1. Run command  `cd /2_5DSEG_PRED`.
#### 2. Change config.yaml file.
#### 1. Run command  `python main_prediction.py`.



# Test results

```
DSC:  0.6808092008717108 +- 0.1749727617385947
Hausdorff:  9.751399615092273 +- 9.279947874717204
Recall:  0.7914909337086632 +- 0.16611485702407042
F1:  0.5269392661792544 +- 0.1548622287067465
AVD:  55.08561295745984 +- 54.99577215339091
```

The other models used:

nnU-Net -> https://catalog.ngc.nvidia.com/orgs/nvidia/resources/nnunet_for_pytorch (Achieved best results)

Deep Bayesian Networks (Hypermapp3r) -> https://hypermapp3r.readthedocs.io/en/latest/
