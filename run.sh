#!/bin/bash

set -ex

dataset=$1
lossarity=$2
losstype=$3
salt=$4

python3 babybeaver.py --dataset $dataset --lossarity $lossarity --losstype $losstype --salt $salt
