"""Module containing functions for working with the Tensorflow."""
import datetime
import os
import time
import json
import numpy
import logging

modlog = logging.getLogger(__name__)

import tensorflow as tf
import matplotlib.pyplot as plt


class ModelConfiguration(object):
    """
    Store the information related to the model and give ability
    to save/load it.

    Attributes:
        classes - list of categories to classify
        dirpath - path to the directory containing the model configuration.
        conf_path - path to the configuration file
        model_path - path to write the model to
        fig_path - path to output model fit figure
        log_path - directory path to write the tensorflow log to
        norm - normalisation factor
        terms - list of words corresponding to input vector

    """
    def __init__(self):
        self.classes = []
        self.dirpath = ''
        self.conf_path = ''
        self.model_path = ''
        self.fig_path = ''
        self.log_path = ''
        self.norm = 0.0
        self.terms = []

    def setinfo(self, outputdir, class_list, normfactor, word_list,
                model_name='classifier_model'):
        """
        Set up the model configuration information.

        Arguments:
            outputdir - where the model information will be saved.
            class_list - list of categories
            normfactor - the normalisation factor
            word_list - list of words which correspond to the input vector
            model_name - the name of the model (default="classifier_model")

        """
        if not isinstance(class_list, list):
            raise TypeError("Class list must be of type list.")
        if not isinstance(word_list, list):
            raise TypeError("Word list must be of type list.")
        if not isinstance(normfactor, (int, float)):
            raise TypeError("normfactor must be a numerical type (int, float).")
        self.classes = class_list
        self.dirpath = os.path.abspath(outputdir)
        self.conf_path = os.path.join(self.dirpath, model_name + '.json')
        self.model_path = os.path.join(self.dirpath, model_name + '.h5')
        self.fig_path = os.path.join(self.dirpath, model_name + '.png')
        logname = datetime.datetime.now().strftime("log-%Y%m%d%H%M%S")
        self.log_path = os.path.join(self.dirpath, logname)

        self.norm = float(normfactor)
        self.terms = word_list

    def save(self):
        """
        Write the configuration to a JSON format file.

        Returns:
            output file path
        """
        info_dict = {'model_path':  self.model_path,
                     'figure_path': self.fig_path,
                     'log_path': self.log_path,
                     'classes': self.classes,
                     'terms': self.terms,
                     'norm': self.norm}
        os.makedirs(self.dirpath, exist_ok=True)
        modlog.info('Directory exists: %s' % self.dirpath)
        with open(self.conf_path, 'w', encoding='utf-8') as fh:
            json.dump(info_dict, fh, ensure_ascii=False, indent=2)
        return self.conf_path

    def load(self, filepath):
        """
        Load the model configuration information from a file.

        """
        dirpath, filename = os.path.split(filepath)
        modname, extn = os.path.splitext(filename)
        with open(filepath, 'r', encoding='utf-8') as fh:
            info_dict = json.load(fh)
        self.setinfo(dirpath,
                     info_dict['classes'],
                     info_dict['norm'],
                     info_dict['terms'],
                     modname)


class ClassifierModel(object):
    """
    Interface to the classifier model. This main reason is to tidy up
    some of the tensorflow interface.

    Attributes:
        n_inputs - number of input features
        n_classes - number of categories
        model - the tensorflow model object

    """
    def __init__(self):
        """
        Create the ClassifierModel class. Define the output paths.

        """
        self.n_inputs = 0
        self.n_classes = 0
        self.model = None
        self._defined = False
        self._trained = False


    def define(self, n_inputs, n_classes):
        """
        Define a Neural Network to do the classifying.

        """
        self.n_inputs = n_inputs
        self.n_classes = n_classes

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32,
                                  activation=tf.nn.relu,
                                  input_shape=(self.n_inputs,)
                                  ),
            tf.keras.layers.Dense(16, activation=tf.nn.relu),
            tf.keras.layers.Dense(self.n_classes, activation=tf.nn.softmax)
        ])
        modlog.debug("Defined model layers.")

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        modlog.debug("Compiled the model.")
        self._defined = True

        self.model.summary()
        modlog.info("Tensorflow neural network defined.")

    def train(self, x, y, logfile, nepochs=40, val_split=0.15):
        """
        Train the neural network.

        Args:
            x - training features
            y - training labels
            logfile - path to write the Tensorboard log to
            nepochs - number of training epochs
            val_split - how much of the data to use as validation data

        """
        if not self._defined:
            raise RuntimeError(
                "Model not yet defined, run ClassifierModel.define()")

        self.epochs = nepochs
        self._tensorboard_callback = \
            tf.keras.callbacks.TensorBoard(log_dir=logfile)
        modlog.debug("Starting model fit.")
        start = time.time()
        hist = self.model.fit(x=x, y=y,
                              epochs=self.epochs,
                              shuffle=True,
                              validation_split=val_split,
                              callbacks=[self._tensorboard_callback],
                              verbose=0
                              )
        end = time.time()
        self._trained = True
        elapsed = end - start
        modlog.info("Training of model complete.")
        modlog.info("Training took %d seconds." % elapsed)

        self._acc = hist.history['acc']
        self._val_acc = hist.history['val_acc']

        self._loss = hist.history['loss']
        self._val_loss = hist.history['val_loss']

    def test(self, x, y):
        """
        Run the model evaluation on the test data.

        Args:
            x - test feature data
            y - test label data

        """
        if not self._trained:
            raise RuntimeError(
                "Model not trained yet. Run ClassifierModel.train().")
        self._test_loss, self._test_acc = self.model.evaluate(x=x, y=y)

    def predict(self, x):
        """
        Make a prediction using the trained model.

        Args:
            x - array of normalised word counts

        Returns:
            list of the probabilities of each category

        """
        if not self._trained:
            raise RuntimeError(
                "Model not trained yet. Run ClassifierModel.train().")
        return self.model.predict(x=x)

    def stats(self):
        """
        Display the final training metrics.

        Returns:
             string containing training performance
        """
        if not self._trained:
            raise RuntimeError(
                "Model not trained yet. Run ClassifierModel.train().")
        output = "\n\tTraining accuracy = %4.3f" % self._acc[-1]
        output += "\n\tValidation accuracy = %4.3f" % self._val_acc[-1]
        output += "\n\tTest accuracy = %4.3f" % self._test_acc
        output += "\n\n\tTraining loss = %4.3f" % self._loss[-1]
        output += "\n\tValidation loss = %4.3f" % self._val_loss[-1]
        output += "\n\tTest loss = %4.3f" % self._test_loss
        return output

    def export_plot(self, figpath):
        """
        Plot the training performance of the model and save it to a file.

        """
        if not self._trained:
            raise RuntimeError(
                "Model not trained yet. Run ClassifierModel.train().")
        epochs_range = range(self.epochs)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, self._acc,
                 label='Training Accuracy (%4.3f)' % self._acc[-1])
        plt.plot(epochs_range, self._val_acc,
                 label='Validation Accuracy (%4.3f)' % self._val_acc[-1])
        plt.plot([epochs_range[0], epochs_range[-1]],
                 [self._test_acc, self._test_acc], 'r--',
                 label='Test Accuracy (%4.3f)' % self._test_acc)
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, self._loss,
                 label='Training Loss (%4.3f)' % self._loss[-1])
        plt.plot(epochs_range, self._val_loss,
                 label='Validation Loss (%4.3f)' % self._val_loss[-1])
        plt.plot([epochs_range[0], epochs_range[-1]],
                 [self._test_loss, self._test_loss], 'r--',
                 label='Test Loss (%4.3f)' % self._test_loss)
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig(figpath)
        modlog.info("Saved plot to file: %s" % figpath)

    def save(self, savepath):
        """Save the model for re-use."""

        if not self._trained:
            raise RuntimeError(
                "Model not trained yet. Run ClassifierModel.train().")
        self.model.save(savepath)
        modlog.info("Saved model to %s" % savepath)

    def load(self, loadpath):
        """Load a pre-trained model."""

        self.model = tf.keras.models.load_model(loadpath)
        self._trained = True
        modlog.info("Loaded model from file.")


def pred2str(predictions, classes, full=True, class_only=False):
    """
    Create a string of the prediction results. Create a list of all
    the classes and their probability. Optionally only the most likely
    result can be returned.

    Args:
        predictions - a list of prediction values
        class - a list of strings giving the classes
        full [optional] - print all of the classes and their predictions
        class_only [optional] - only print the most likely class name

    Returns - a string of the predictions.

    """
    if (not isinstance(predictions, (list, numpy.ndarray)) or
        not all(numpy.isreal(predictions))):
        raise TypeError("predictions must be a numeric list.")

    if not isinstance(classes, list):
        raise TypeError("classes must be a list.")

    if not len(predictions) == len(classes):
        raise IndexError("Input variables must be equal lengths.")

    if full and class_only:
        modlog.warning("Options full and class_only can't both be true."
                       "The full option takes precedence.")

    output = ["%s : %4.3f" % (cls, pred)
              for cls, pred in zip(classes, predictions)]
    output = sorted(output, key=lambda x: float(x.split(':')[1]))
    if full:
        return "\n".join(output)
    elif class_only:
        return output[-1].split(' ')[0]
    else:
        return output[-1]
