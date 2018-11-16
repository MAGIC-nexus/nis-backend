#!/bin/bash

# Change this path to use the system python installation (simply "python" or /usr/bin/python)
# or use a specific python environment installation
PYTHON=/home/marco/miniconda3/envs/nis-backend/bin/python

# Get the current GIT branch of the project
CURRENT_BRANCH=`git rev-parse --abbrev-ref HEAD`

if [[ "$CURRENT_BRANCH" == "master" || "$CURRENT_BRANCH" == "develop" ]]; then
  echo "You are on branch >$CURRENT_BRANCH<. Are you sure you want to commit to this branch?"
  echo "If so, commit with option --no-verify to bypass this pre-commit hook."
  exit 1
fi

echo "Running unit tests before committing..."
$PYTHON -m unittest -v backend_tests/test_unit_*.py