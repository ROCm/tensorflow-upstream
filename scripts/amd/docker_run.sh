set -o xtrace

alias drun='sudo docker run -it --rm --network=host --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined'

DEVICES="--device=/dev/kfd --device=/dev/dri"
# DEVICES="--gpus all"

MEMORY="--ipc=host --shm-size 16G"

VOLUMES="-v $HOME/dockerx:/dockerx -v /data:/data"

# WORK_DIR='/dockerx/tensorflow-upstream'
# WORK_DIR='/root/tensorflow-upstream'
# WORK_DIR='/dockerx'
WORK_DIR='/tensorflow-upstream'

# IMAGE_NAME=rocm/tensorflow
IMAGE_NAME=rocm/tensorflow-autobuilds:latest
# IMAGE_NAME=tensorflow/tensorflow:devel-gpu

CONTAINER_ID=$(drun -d -w $WORK_DIR $MEMORY $VOLUMES $DEVICES $IMAGE_NAME)
# CONTAINER_ID=$(drun -d $MEMORY $VOLUMES $DEVICES $IMAGE_NAME)
echo "CONTAINER_ID: $CONTAINER_ID"
docker cp . $CONTAINER_ID:$WORK_DIR
docker attach $CONTAINER_ID
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID
