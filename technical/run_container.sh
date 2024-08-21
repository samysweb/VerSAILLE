#!/bin/bash

mkdir -p $(pwd)/versaille_docker_data

docker run --rm -p 8888:8888  start-notebook.py --NotebookApp.token='versaille'