
# ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.11-py3
# FROM ${FROM_IMAGE_NAME} 
FROM nvcr.io/nvidia/pytorch:22.06-py3

ADD ./requirements.txt .
RUN pip install --disable-pip-version-check -r requirements.txt
RUN pip install monai==0.8.0 --no-dependencies
RUN pip uninstall -y torchtext

RUN pip install torchmetrics==0.6.0

WORKDIR /workspace/nnunet_pyt
ADD main.py /workspace/nnunet_pyt
