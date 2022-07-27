FROM projectmonai/monai:0.9.1rc3

RUN pip install hydra-core==1.1.1
RUN pip install omegaconf==2.1.1
RUN pip install SimpleITK==2.1.1
RUN pip install monai==0.8.0
RUN pip install rich==10.16.1