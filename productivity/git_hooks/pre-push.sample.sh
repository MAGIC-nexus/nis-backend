#!/bin/bash

# Change this path to use the system python installation (simply "python" or /usr/bin/python)
# or use a specific python environment installation
PYTHON=/home/marco/miniconda3/envs/nis-backend/bin/python

# Get the current GIT branch of the project
CURRENT_BRANCH=`git rev-parse --abbrev-ref HEAD`

if [[ "$CURRENT_BRANCH" == "master" || "$CURRENT_BRANCH" == "develop" ]]; then
  echo "Running integration tests before pushing to develop or master..."
  $PYTHON -m unittest -v backend_tests/test_integration_*.py
fi
