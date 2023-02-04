#!/bin/bash

poetry export --without-hashes --dev -f requirements.txt > requirements.txt
