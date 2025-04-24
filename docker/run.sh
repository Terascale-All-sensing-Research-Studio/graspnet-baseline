#!/bin/bash

# Run
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    --ipc=host \
    --volume /media:/media \
    graspnet-baseline \
    bash
