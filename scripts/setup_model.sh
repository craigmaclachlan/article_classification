#!/bin/bash
#
# Run the scripts to retrieve and process data, then train a model
#
# Usage: setup_model.sh
#

scripts_dir=$(dirname $0)

${scripts_dir}/get_training_data.sh $PWD/training_data

${scripts_dir}/process_training_data.py $PWD/training_data $PWD/training_data/articles.csv

${scripts_dir}/build_model.py $PWD/training_data/articles.csv \
                              $PWD/training_data/bbc.classes \
                              $PWD/model