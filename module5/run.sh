#!/usr/bin/env bash

NUM_THREADS="$1"
BLOCK_SIZE="$2"

./build/assignment.exe $NUM_THREADS $BLOCK_SIZE
