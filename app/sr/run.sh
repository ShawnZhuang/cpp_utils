#!/bin/bash

# Add Python path
SCRIPT_DIR=$(realpath $(dirname "${BASH_SOURCE[0]}"))
ROOT_DIR=$(realpath "${SCRIPT_DIR}/../..")
echo ${SCRIPT_DIR}
export PYTHONPATH="${ROOT_DIR}/python:${SCRIPT_DIR}:$PYTHONPATH"

python3 app/sr/run.py
# Your script commands go here