#!/bin/bash

script=$1
config=$2

source $config
python3 $script ${gpt_options}
