set -o xtrace

CONTAINER_NAME=flamboyant_hamilton


WORK_DIR='/tensorflow-upstream'

SRC=.
# SRC=test
# SRC=scripts

docker cp $SRC $CONTAINER_NAME:$WORK_DIR