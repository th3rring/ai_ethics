#! /bin/bash

# tmux new-session -d -s "jupyter" jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token='robobio'
# jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token='robobio'
# jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token='robobio'

$CONDA_PREFIX/bin/jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token='robobio'
