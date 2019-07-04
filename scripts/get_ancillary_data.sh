#!/bin/bash


output_dir=$1
if [[ ! -d ${output_dir} ]]; then
    mkdir -p ${output_dir}
fi

data_url="http://mlg.ucd.ie/files/datasets/stopwords.txt"
filename=$(basename ${data_url})

curl -o ${output_dir}/${filename} ${data_url}
if [[ $? -eq 0 ]]; then
    echo "$0: Successfully downloaded data: ${data_url} to ${output_dir}"
else
    echo "$0: ERROR: Data download failed."
    exit 1
fi

python -c "import nltk; nltk.download('punkt')"
if [[ $? -eq 0 ]]; then
    echo "$0: Successfully installed data for NLTK"
else
    echo "$0: ERROR: Unable to download NLTK data (punkt)."
    exit 1
fi