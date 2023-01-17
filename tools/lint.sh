#!/bin/sh

python -m black **/*.py
python -m isort --profile black **/*.py
