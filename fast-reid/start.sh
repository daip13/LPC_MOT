CKER_ADDRESS=registry.aibee.cn/aibee/torchpp:1.4.2
docker pull $DOCKER_ADDRESS
docker run --shm-size=128gb -it -d \
    --network=host \
    -e COLUMNS=`tput cols` \
    -e LINES=`tput lines` \
    -v /etc/localtime:/etc/localtime:ro \
    -v /ssd:/ssd \
    -v /mnt:/mnt \
    -v $PWD/..:/workspace \
    $DOCKER_ADDRESS \
    bash

