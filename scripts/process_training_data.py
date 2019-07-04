#!/usr/bin/env python3
"""
Process the input data from the BBC into a format that will be easy to
use in Tensorflow.

"""
import argparse
#import logging
import logging.config

import articlass

logging.config.dictConfig(articlass.logging_config)
logger = logging.getLogger()
logger.debug("Logger set up.")

import articlass.process
import articlass.model

parser = argparse.ArgumentParser(
    description='Process the input data from the BBC into a format (CSV) '
                'that will be easy to use in Tensorflow.')
parser.add_argument('input_dir',
                    help='Directory containing the input data files '
                         '(bbc.classes, bbc.terms, bbc.mtx).')
parser.add_argument('output_path',
                    help='Path to write the CSV data to.')

args = parser.parse_args()
logger.debug(
    "Read arguments: input dir: %s, output_path: %s" % (args.input_dir,
                                                        args.output_path))

# Create the training data class and load in the data.
training_data = articlass.process.TrainingData()
training_data.load_raw(args.input_dir)

# Write out the training data
training_data.export(args.output_path)
