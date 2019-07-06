#!/usr/bin/env python3
"""
Build a classifier model trained on the article data.

"""
import argparse
import logging.config

import articlass

# TODO: Tensorflow is mucking up the logging. Needs fixing
logging.config.dictConfig(articlass.logging_config)
logger = logging.getLogger(__name__)
logger.debug("Logger set up.")

import articlass.model
import articlass.process


parser = argparse.ArgumentParser(
    description='Build a classifier model trained on the article data.')
parser.add_argument('input_path',
                    help='Path to CSV file containing the word counts '
                         'and labels.')
parser.add_argument('class_file',
                    help='Path to file containing the class information.')
parser.add_argument('output_path',
                    help='Directory path to save the model information to.')

args = parser.parse_args()

#
# Set up the training data.
#
training_data = articlass.process.TrainingData()
training_data.load_csv(args.input_path)

train_x, train_y, test_x, test_y = \
    training_data.train_test_split(test_frac=0.15)

train_x = train_x / training_data.feature_max_value
test_x = test_x / training_data.feature_max_value
logger.info("Normalised the feature arrays.")

# Set up the model.
class_info = articlass.process.ClassFile(args.class_file)
class_info.get_classes()

model_cfg = articlass.model.ModelConfiguration()
model_cfg.setinfo(args.output_path,
                  class_info.classes,
                  training_data.feature_max_value,
                  training_data.feature_names)

nn_model = articlass.model.ClassifierModel()
#
# Build the model
#
nn_model.define(len(training_data.feature_names), class_info.number)

# De-reference the data we loaded into the pandas DataFrame to
# reduce the memory footprint
training_data.clean()

# Train the model and evaluate test performance
nn_model.train(train_x, train_y, model_cfg.log_path)
nn_model.test(test_x, test_y)

# Output performance statistics
logger.info(nn_model.stats())
nn_model.export_plot(model_cfg.fig_path)

# Save the model and config so that it can be used again later.
model_cfg.save()
nn_model.save(model_cfg.model_path)
