#!/bin/bash

if [[ "$(docker images -q myjupyterlab-p8 2> /dev/null)" == "" ]]; then
    ./build_docker.sh
fi

mkdir -p /home/nordine/Projets/DataScience/DataScience_Training/.cache-projet8

docker run --rm -v /home/nordine/Projets/DataScience/DataScience_Training/projet8:/home/jupyter/notebooks \
	-v /home/nordine/Projets/DataScience/DataScience_Training/.cache-projet8:/home/jupyter/.cache  \
	--network host --gpus all myjupyterlab-p8

