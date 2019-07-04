#!/bin/bash
#
# This script retrieves and unpacks the training data for the model we will build.
# The data is described here: http://mlg.ucd.ie/datasets/bbc.html
#
# Usage: get_training_data.sh <output_dir>
# Args: output_dir - where to store the data we download

output_dir=$1
if [[ ! -d ${output_dir} ]]; then
    mkdir -p ${output_dir}
fi

data_url="http://mlg.ucd.ie/files/datasets/bbc.zip"
filename=$(basename data_url)


curl -o ${output_dir}/${filename} ${data_url}
if [[ $? -eq 0 ]]; then
    echo "$0: Successfully downloaded data: ${data_url} to ${output_dir}"
else
    echo "$0: ERROR: Data download failed."
    exit 1
fi

unzip -o ${output_dir}/${filename} -d ${output_dir}
if [[ $? -eq 0 ]]; then
    echo "$0: Successfully decompressed data: ${output_dir}/${filename}"
else
    echo "$0: ERROR: Data unzip failed."
    exit 1
fi

# Check that we have the files we are expecting
expected="bbc.classes bbc.docs bbc.mtx bbc.terms"
for f in ${expected}; do
    if [[ ! -f ${output_dir}/${f} ]]; then
        echo "$0: ERROR: Required file is missing: ${f}"
        exit 1
    fi
done
echo "$0: Success. All required files present."