#!/usr/bin/env python3
"""Use a pre-trained model to classify a BBC article."""

import argparse
import requests
import collections
import numpy
import nltk.tokenize
import nltk.stem
import logging.config

import articlass
# TODO: Tensorflow is mucking up the logging. Needs fixing
logging.config.dictConfig(articlass.logging_config)
logger = logging.getLogger()
logger.debug("Logger set up.")

import articlass.text
import articlass.model
import articlass.process

parser = argparse.ArgumentParser(
    description='Use a pre-trained model to classify a BBC article.')
parser.add_argument('model_config',
                    help='Path to the model configuration file. This file '
                         'contains all the information to apply the model.')
parser.add_argument('url',
                    help='URL of a BBC article')

args = parser.parse_args()

# Retrieve the data from the URL
logger.info("Retrieving data from URL: %s" % args.url)
response = requests.get(args.url)
logger.info("Data retrieved from URL, status: %s" % response.status_code)
logger.debug("Response encoding: %s" % response.encoding)

# Convert the HTML into plain text.
text = articlass.text.html2text(response.text)
logger.info("Stripped HTML syntax.")

# Convert the text block into a list of words.
word_list = nltk.tokenize.word_tokenize(text)
logger.info("Number of words in article: %d" % len(word_list))

porter = nltk.stem.PorterStemmer()
countable_words = [porter.stem(w) for w in word_list]
logger.info("Number of countable words = %d" % len(word_list))

# The Counter class from collections is essentially a dictionary
# where the values are the counts of each key in the original list.
counts = collections.Counter(countable_words)


model_cfg = articlass.model.ModelConfiguration()
model_cfg.load(args.model_config)

# The words that the model has been trained on are pre-defined
# and need to be in a specific order. Go through all of the terms
# and count how many were in the article.

# Calculate the input for the model - counts of the pre-defined terms.
model_input = articlass.text.count_terms(counts, model_cfg.terms)

# Load the model
nn_model = articlass.model.ClassifierModel()
nn_model.load(model_cfg.model_path)

# Convert the counts list to a numpy array and normalise it. The
# normalisation factor needs to be the one calculated when the model
# was trained.
# The input data needs to be a list of input vectors, so reshape it.
model_input = numpy.array(model_input).reshape(1, len(model_input))
model_input = model_input / model_cfg.norm

# Make the prediction
predictions = nn_model.predict(x=model_input)[0]

logger.info(
    articlass.model.pred2str(predictions, model_cfg.classes, full=True))

# Here we print the result of the classification to stdout because this is
# the result of the programme. Don't use the logger because that might be
# redirected to a file
print("%s : %s" % (
    args.url,
    articlass.model.pred2str(predictions, model_cfg.classes,
                             full=False, class_only=True)
    ))
