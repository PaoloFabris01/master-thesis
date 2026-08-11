#FROM nvcr.io/nvidia/pytorch:25.05-py3
FROM python:3.12
ARG USER=standard
ARG USER_ID=1007 # uid from the previus step
ARG USER_GROUP=standard
ARG USER_GROUP_ID=1007 # gid from the previus step
ARG USER_HOME=/home/${USER}
# create a user group and a user (this works only for debian based images)
RUN groupadd --gid $USER_GROUP_ID $USER \
    && useradd --uid $USER_ID --gid $USER_GROUP_ID -m $USER
# setup image istructions
RUN apt-get update && apt-get install -y curl

# add libraries to install
RUN python -m pip install --upgrade pip
#dependencies for gdal
#RUN apt-get install -y libgdal-dev gdal-bin g++ --no-install-recommends && \
#    apt-get clean -y

## Install the application dependencies
RUN pip install numpy
RUN pip install torch
RUN pip install torchvision
RUN pip install torch_geometric
RUN pip install imageio
RUN pip install matplotlib
RUN pip install pandas
RUN pip install scipy
RUN pip install keras
RUN pip install scikit-image
RUN pip install scikit-learn
RUN pip install geopandas
RUN pip install opencv-python
RUN pip install tqdm
RUN pip install rasterio
RUN pip install networkx

# set container user
USER $USER
# run script as non-root user
ENTRYPOINT ["python", "model_slic_on_each_time_patch_3.py"]



