#!/bin/bash

if [[ "$(docker images -q myjupyterlab_p7 2> /dev/null)" == "" ]]; then
    ./build_docker.sh
fi

mkdir -p /home/nordine/Projets/DataScience/DataScience_Training/.cache-projet7

docker run --rm -v /home/nordine/Projets/DataScience/DataScience_Training/projet7:/home/jupyter/notebooks \
	-v /home/nordine/Projets/DataScience/DS_P7_API:/home/jupyter/DS_P7_API \
	-v /home/nordine/Projets/DataScience/DS_P7_FRONTEND:/home/jupyter/DS_P7_FRONTEND \
	-v /home/nordine/Projets/DataScience/DataScience_Training/.cache-projet7:/home/jupyter/.cache  \
	--network host --gpus all myjupyterlab_p7

