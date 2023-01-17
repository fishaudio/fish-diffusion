#!/bin/sh

python -m black **/*.py *.py
python -m isort --profile black **/*.py *.py
