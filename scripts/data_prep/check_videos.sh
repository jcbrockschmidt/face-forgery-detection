#!/usr/bin/env bash

for VIDEO in "$@"
do
    echo "Checking $VIDEO..."
    ffmpeg -v error -i $VIDEO -f null -
done
