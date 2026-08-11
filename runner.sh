#!/bin/bash

GPU_ID=3;

docker run --rm --runtime=nvidia --name='graph-pfabris-rsde-'${GPU_ID} -e CUDA_VISIBLE_DEVICES=$GPU_ID --ipc=host \
--ulimit memlock=-1 --ulimit stack=67108864 -t --rm -u $(id -u):$(id -g) -v $(pwd):$(pwd) -w $(pwd) graph-pfabris-rsde
