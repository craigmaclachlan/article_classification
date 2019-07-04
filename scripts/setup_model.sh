#!/bin/bash

get_training_data.sh $PWD/training_data

get_ancillary_data.sh $PWD/ancil

process_training_data.py $PWD/training_data $PWD/training_data/articles.csv

build_model.py $PWD/training_data/articles.csv $PWD/model