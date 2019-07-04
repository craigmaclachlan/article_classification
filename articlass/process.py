"""Functions for processing the data."""

import scipy.io
import pandas
import numpy
import logging
import os.path
import sklearn.model_selection

import articlass.text

modlog = logging.getLogger(__name__)


class TrainingData(object):
    """
    This class is responsible for loading, transforming and exporting
    training data to be used in training a Tensorflow model.

    It can also load pre-processed data.

    The model will classify BBC articles based on the text, specifically the
    frequency of a pre-defined set of vocabulary.

    TODO: Add a sanity checking method. Maybe plot/print some stats.

    Attributes:
        dataframe - pandas dataframe containing the words counts and categories
        norm_factor - the maximum value of the word counts in the dataframe.
        feature_max_value - the maximum value of all the word counts.

    """
    def __int__(self):
        self.dataframe = None
        self.norm_factor = 0

    def load_raw(self, datapath):
        """
        Create the TrainingData object. Requires the path to a set of files:
        * bbc.classes - list of the categories for each article.
        * bbc.mtx - sparse matrix of the word counts for all of the articles
        * bbc.terms - list of the pre-defined vocabulary.

        Args:
            datapath - path to a directory containing the above files.

        """
        classes = ClassFile(os.path.join(datapath, "bbc.classes"))
        counts = WordCountFile(os.path.join(datapath, "bbc.mtx"))
        terms = TermsFile(os.path.join(datapath, "bbc.terms"))

        # Read in all the data.
        classes.read()
        modlog.info("Loaded class information.")
        modlog.info(str(classes))

        counts.read()
        modlog.info("Loaded word counts.")

        terms.read()
        modlog.info("Loaded terms.")

        # Consistency checks
        if not classes.n_samples == counts.n_samples:
            raise IndexError(
                "Number of samples not consistent between bbc.classes and"
                "bbc.mtx: %d v %d" % (classes.n_samples, counts.n_samples))

        if not counts.n_terms == terms.n_terms:
            raise IndexError(
                "Number of terms not consistent between bbc.mtx and"
                "bbc.terms: %d v %d" % (counts.n_terms, terms.n_terms))

        self.dataframe = pandas.DataFrame(counts.array, columns=terms.words)
        self.dataframe['_labels_class_'] = classes.labels

    def load_csv(self, filepath):
        """
        Load a pre-processed set of training data from a CSV file.

        Args:
            filepath - path to a CSV file containing pre-processed data

        """
        self.dataframe = pandas.read_csv(filepath)
        modlog.info("Loaded CSV data from: %s" % filepath)

    @property
    def feature_max_value(self):
        """Calculate the maximum value of all the features."""
        if self.dataframe is None:
            raise ValueError(
                "Data must be loaded before max value can be calculated.")
        return self.dataframe.max().drop(columns=['_labels_class_']).max()

    @property
    def feature_names(self):
        """List of words that comprise the features."""
        feature_list = self.dataframe.columns.to_list()
        feature_list.remove('_labels_class_')
        return feature_list

    def export(self, filepath):
        """
        Export the training dataframe to a CSV file.

        Args:
            filepath - path to export the dataframe to.

        """
        # The dataframe index isn't saved here because we don't really
        # want it and it can cause problems when reading the data back in.
        self.dataframe.to_csv(filepath, index=False)
        modlog.info("DataFrame written to file: %s" % filepath)

    def train_test_split(self, test_frac=0.15):
        """
        Split the data into training and test (numpy) arrays.

        Args:
            test_frac - the fraction of the training data to set aside
                        for the test set

        Returns:
            tuple of:
             (training features,
              training labels,
              test features,
              test labels)

        """
        labels = self.dataframe['_labels_class_']
        features = self.dataframe.drop(columns=['_labels_class_'])
        modlog.debug("Set up labels and features.")

        train_x, test_x, train_y, test_y = \
            sklearn.model_selection.train_test_split(features,
                                                     labels,
                                                     test_size=test_frac,
                                                     shuffle=True)
        modlog.info("Split training data (ratio: %4.3f)" % test_frac)
        train_x = train_x.values
        test_x = test_x.values
        modlog.debug("Extracted values from dataframes.")
        train_y = numpy.array(train_y)
        test_y = numpy.array(test_y)
        modlog.debug("Converted the label lists to numpy arrays.")

        ncols, nsamples = train_x.shape
        ntestsamples = len(test_y)
        modlog.info("Number of input columns: %d" % ncols)
        modlog.info("Training samples: %d" % nsamples)
        modlog.info("Test samples: %d" % ntestsamples)
        return train_x, train_y, test_x, test_y

    def clean(self):
        """
        Training the model can be a memory intensive process, so this
        method sets the dataframe to None. Hopefully python
        garbage collection recovers the memory.

        This strategy is untested.

        """
        self.dataframe = None


class ClassFile(object):
    """
    Interface to a class file. The header contains information on the
    categories and the body is a list of category indices defining the
    category of article in the dataset.

    Attributes:
        filepath - path to the class file
        classes - list of the categories [strings]
        number - the number of classes
        n_samples - the number of entries in the dataset
        labels - list of the category for each article

    """
    def __init__(self, filepath):
        """
        Interface to the class information in the dataset.

        Args:
            filepath - path to the class file

        """
        if not os.path.exists(filepath):
            raise IOError("File does not exist: %s" % filepath)
        self.filepath = filepath
        self.classes = '?'
        self.number = 0
        self.n_samples = 0
        self.labels = []

    def __repr__(self):
        return "%d categories: %s; %d samples" % (self.number,
                                                  ", ".join(self.classes),
                                                  self.n_samples)

    def get_classes(self):
        """
        Extract the classes from the class file and put in a list.

        The header of this file contains the list of the classes. The body
        is a list of the classes for each training sample.

        The key line looks like this:
        %Clusters 5 business,entertainment,politics,sport,tech

        Returns:
            names of classes as a list of strings

        """
        classes = ''
        with open(self.filepath, 'r') as fh:
            modlog.info("Opened file: %s" % self.filepath)
            for line in fh.readlines():
                if "Clusters" in line:
                    splitline = line.split(" ")
                    classes = splitline[-1]
                    self.number = int(splitline[1])
                    modlog.debug("Found classes: %s" % classes)
                    break
        if classes == '':
            raise RuntimeError("Unable to find class information.")
        class_list = [cls.strip() for cls in classes.split(",")]
        if not len(class_list) == self.number:
            modlog.debug("Length class list: %d" % len(class_list))
            modlog.debug("Defined number of Classes : %d" % self.number)
            raise ValueError("Number of classes not correctly defined in file.")
        self.classes = class_list
        modlog.info("Loaded class information.")

    def read(self):
        """
        Read the file and get the list of article categories.

        """
        if self.classes == '?':
            self.get_classes()
        labels = pandas.read_csv(self.filepath, sep=" ", dtype=int,
                                 names=['Name', 'NaturalClasses'], comment='%')
        self.labels = labels['NaturalClasses'].values
        self.n_samples = len(self.labels)
        modlog.info("Loaded Class information.")


class WordCountFile(object):
    """
    Interface to a file containing the word counts for a series of articles.

    Attributes:
        filepath - path to the word count file
        array - the non-sparse array of the word counts [numpy.array]
        n_samples - the number of samples (articles) in the dataset
        n_terms - the number of words that have been counted.

    """
    def __init__(self, filepath):
        """
         Interface to the word count information in the dataset.

         Args:
             filepath - path to the word count file

        """
        if not os.path.exists(filepath):
            raise IOError("File does not exist: %s" % filepath)
        self.filepath = filepath
        self.array = None
        self.n_samples = 0
        self.n_terms = 0

    def read(self):
        """
        Read the word counts from a file in Matrix Market format. The matrix
        gets transposed so that each row is an article and the columns are
        the word counts.

        """
        sp_array = scipy.io.mmread(self.filepath)
        modlog.info("Read file: %s" % self.filepath)

        self.array = numpy.transpose(sp_array.todense())
        self.n_samples, self.n_terms = self.array.shape


class TermsFile(object):
    """
    The terms file contains a list of predefined words. These are the
    words that have been counted in the word count file.

    Attributes:
        filepath - path to the terms file
        words - list of the words [strings]
        n_terms - number of words

    """
    def __init__(self, filepath):
        """
         Interface to the terms information in the dataset.

         Args:
             filepath - path to the terms file

        """
        if not os.path.exists(filepath):
            raise IOError("File does not exist: %s" % filepath)
        self.filepath = filepath
        self.words = []
        self.n_terms = 0

    def read(self):
        """
        Read the list of predefined words that have been counted to
        create this dataset.

        """
        self.words = articlass.text.load_words(self.filepath)
        modlog.info("Read file: %s" % self.filepath)
        self.n_terms = len(self.words)
