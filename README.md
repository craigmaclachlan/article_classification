# ArtiClass: Article classifying

This package automates the building of a neural network 
classifier and allows article URLs to be classified.

## Overview
There are two main parts to this package:
1. The retrieval and processing of data. Followed by the training of a
neural network classifier. This whole process can be executed by 
running: `setup_model.sh`. 
2. An application which takes a URL, processes the text and then makes
a prediction of the class of the article using the trained model. The
application is `classify_url.py`.

## Installation
The package has various dependencies. The easiest way to install these is 
by using the following command:

`python setup.py install`

## Training Data
The training data is a dataset made available by the BBC, and can be found here: 
http://mlg.ucd.ie/datasets/bbc.html

There are two datasets: general news articles and sport news articles. Both 
datasets have been processed in the same way: stemming, stop-word removal,
and low frequency term filtering (less than 3). The news article 
dataset is comprised of several files, the key ones are:
* Term frequencies (*.mtx)
* List of terms (*.terms) that have been counted.
* List of article classes, or categories (*.classes)

The script `scripts/get_training_data.sh` automates the retrieval of 
the data.

## Processing data
For use in [Tensorflow](https://www.tensorflow.org/) the preprocessed data is 
processed into a CSV file using the [pandas library](https://pandas.pydata.org/).

The word count is an sparse matrix format, this is converted to a dense numeric
array with column names relating to the predefined terms. A column of the 
categories is added to the table. This table is exported to CSV and contains 
all of the relevant information.

The processing of the data can be run like this:

`scripts/process_training_data.py input_data_path outfile.csv`

## Classifier model
A simple neural network is used to classify the articles. The model has
2 hidden layers of 32 and 16 nodes. The accuracy of the trained model
is greater than 95%.

The model is trained by running:

`scripts/build_model.py outfile.csv training_file.classes model_output_dir`

The model can be altered by modifying the `ClassifierModel.define()` 
method in the `articlass.model` file.

In the training of the model several diagnostics are produced:
* The script will display the accuracy and loss of the training, 
validation and test subsets of the data. 
* The [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard)
visualisation is recorded during training and can be viewed by running:

  `tensorboard --logdir=model_output_dir/log-YYYYMMDDHHMMSS`
* A static plot showing the evolution of the training accuracy and loss
is stored in the model output directory. This presents similar 
information to the tensorboard output, but does not require the 
tensorboard application to be run.  

The model is saved to file along with a configuration file which
contains all the information required to load and apply the model.
The configuration file (JSON format) contains paths to the model save,  
paths to the log outputs, a word count normalisation factor, list of categories, 
and the list of words that need to be counted to create the input to the model.


## Classifier Application
The pre-trained model can be applied to new data using the script:

`scripts/classify_url.py model_conf.json https://www.bbc.co.uk/news/uk-scotland-edinburgh-east-fife-48885287
`

The script has two main parts:
#### Text processing
Retrieve the data from the URL and convert it to neural network model input 
data. The html source of the URL is retrieved and the HTML tags are filtered
out to leave the article text. The article text is then "stemmed"; this is the
process of removing the endings of words to leave the "root" of a word. For 
example "important" and "importance" should resolve to the same root.

Filtering of "stopwords" (common words not to count) is applied to the 
training dataset. However, this is not applied to the prediction of new data.
It is crucial that the words counted here are exactly the same set as the model
was trained on, otherwise the model will the work.

The low frequency words (1 or 2 instances) are removed because the model has 
not been exposed to these values, and may give unexpected results when given 
these values.
 
#### Apply the model
The count of the relevant words in the article is passed to the pre-trained 
model. The model returns the probability of the article being in each class.
The full list of probabilities is written to the log (default stdout). The 
article URL and the most likely class is always printed to stdout. For example:
>  tech : 0.039\
>  sport : 0.168\
>  politics : 0.299\
>  entertainment : 0.485\
>  https://www.bbc.co.uk/news/uk-scotland-edinburgh-east-fife-48885287 : entertainment


## Future Work


