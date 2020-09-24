#!/usr/bin/env bash

DOCKER_ADDRESS=registry.aibee.cn/aibee/torchpp:1.4.2
docker pull $DOCKER_ADDRESS
nvidia-docker run --shm-size=128gb -it --rm -d \
    --network=host \
    -e COLUMNS=`tput cols` \
    -e LINES=`tput lines` \
    -v /etc/localtime:/etc/localtime:ro \
    -v /ssd:/ssd \
    -v /mnt:/mnt \
    -v $PWD/..:/workspace \
    -p 12355:12355 \
    $DOCKER_ADDRESS \
    bash
