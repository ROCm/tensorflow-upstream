set -o xtrace

alias drun='sudo docker run -it --rm --network=host --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined'

# DEVICES="--gpus all"
# DEVICES="--device=/dev/kfd --device=/dev/dri"

MEMORY="--ipc=host --shm-size 16G"

DIR_NAME="$(dirname $(pwd))"
VOLUMES="-v $DIR_NAME:/dockerx -v /data:/data"

WORK_DIR="/dockerx/$(basename $(pwd))"
# WORK_DIR="/root/$(basename $(pwd))"

# IMAGE_NAME=rocm/tensorflow
IMAGE_NAME=rocm/tensorflow-autobuilds:latest
# IMAGE_NAME=tensorflow/tensorflow:devel-gpu
# IMAGE_NAME=rocm/tensorflow-private:enable_rccl_debug_build

CONTAINER_NAME=tensorflow

# start new container
CONTAINER_ID=$(drun -d -w $WORK_DIR --name $CONTAINER_NAME $MEMORY $VOLUMES $DEVICES $IMAGE_NAME)
# docker cp . $CONTAINER_ID:$WORK_DIR
# docker exec $CONTAINER_ID bash -c "bash scripts/amd/run.sh"
docker attach $CONTAINER_ID
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID
