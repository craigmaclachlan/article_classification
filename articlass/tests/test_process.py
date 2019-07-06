import os
import shutil
import unittest
import filecmp
import tempfile
import numpy
import pandas
import logging
#import logging.config

#import articlass
#logging.config.dictConfig(articlass.logging_config)
#logger = logging.getLogger()

from articlass.process import TrainingData, ClassFile, WordCountFile, TermsFile


class TestTrainingData(unittest.TestCase):
    """Test the TrainingData class."""

    def setUp(self):
        self.dirpath = tempfile.mkdtemp()

        self.classfile = os.path.join(self.dirpath, 'bbc.classes')
        with open(self.classfile, 'w') as fh:
            fh.write("% This is a file\n%Clusters 3 cat,dog,bear\n% foobar\n")
            fh.write("0 0\n1 1\n2 1")

        self.mtxfile = os.path.join(self.dirpath, 'bbc.mtx')
        with open(self.mtxfile, 'w') as fh:
            mmf = """%%MatrixMarket matrix coordinate real general
            2 3 4
            1 1 1.0
            1 3 5.0
            2 1 2.0
            2 2 2.0"""
            fh.write(mmf)

        self.termsfile = os.path.join(self.dirpath, 'bbc.terms')
        with open(self.termsfile, 'w') as fh:
            fh.write("fluffy\nscary\n")

        self.csvfile = os.path.join(self.dirpath, 'test.csv')
        with open(self.csvfile, 'w') as fh:
            fh.write("fluffy,scary,_labels_class_\n")
            fh.write("1.0,2.0,0\n")
            fh.write("0.0,2.0,1\n")
            fh.write("5.0,0.0,1\n")

        self.expected_df = pandas.DataFrame(
            [[1.0, 2.0, 0], [0.0, 2.0, 1], [5.0, 0, 1]],
            columns=["fluffy", "scary", "_labels_class_"])

    def tearDown(self):
        shutil.rmtree(self.dirpath)

    def test_good_case(self):

        td = TrainingData()
        td.load_raw(self.dirpath)
        self.assertTrue(self.expected_df.equals(td.dataframe))

    def test_missing_file(self):
        os.remove(self.mtxfile)
        with self.assertRaises(IOError):
            td = TrainingData()
            td.load_raw(self.dirpath)

    def test_inconsistent_terms(self):
        with open(self.termsfile, 'w') as fh:
            fh.write("fluffy\nscary\ntoothy\n")
        with self.assertRaises(IndexError):
            td = TrainingData()
            td.load_raw(self.dirpath)

    def test_inconsistent_samples(self):
        with open(self.classfile, 'w') as fh:
            fh.write("% This is a file\n%Clusters 3 cat,dog,bear\n% foobar\n")
            fh.write("0 0\n1 1\n2 1\n3 3")
        with self.assertRaises(IndexError):
            td = TrainingData()
            td.load_raw(self.dirpath)

    def test_max_value(self):
        td = TrainingData()
        td.load_raw(self.dirpath)
        self.assertEqual(5, td.feature_max_value)

    def test_feature_names(self):
        td = TrainingData()
        td.load_raw(self.dirpath)
        self.assertEqual(['fluffy', 'scary'], td.feature_names)

    def test_load_csv(self):
        td = TrainingData()
        td.load_csv(self.csvfile)
        self.assertTrue(self.expected_df.equals(td.dataframe))

    def test_export_csv(self):
        export_file = os.path.join(self.dirpath, 'export.csv')
        td = TrainingData()
        td.load_raw(self.dirpath)
        td.export(export_file)
        self.assertTrue(filecmp.cmp(self.csvfile, export_file))

    def test_clean(self):
        td = TrainingData()
        td.load_raw(self.dirpath)
        td.clean()
        self.assertIsNone(td.dataframe)

    def test_spit(self):
        exp_trainx = numpy.array([[1.0, 2.0], [0.0, 2.0]])
        exp_trainy = numpy.array([0, 1])
        exp_testx = numpy.array([[5.0, 0]])
        exp_testy = numpy.array([1])

        td = TrainingData()
        td.load_raw(self.dirpath)

        trainx, trainy, testx, testy = td.train_test_split(test_frac=0.33,
                                                           shuffle=False)

        self.assertTrue(numpy.array_equal(exp_trainx, trainx))
        self.assertTrue(numpy.array_equal(exp_trainy, trainy))
        self.assertTrue(numpy.array_equal(exp_testx, testx))
        self.assertTrue(numpy.array_equal(exp_testy, testy))

    def test_split_not_fraction(self):
        td = TrainingData()
        td.load_raw(self.dirpath)

        with self.assertRaises(ValueError):
            td.train_test_split(test_frac=1, shuffle=False)

    def test_split_negative(self):
        td = TrainingData()
        td.load_raw(self.dirpath)

        with self.assertRaises(ValueError):
            td.train_test_split(test_frac=-0.5, shuffle=False)


class TestClassFileHeader(unittest.TestCase):
    """Test the extraction of class information from the header."""
    def setUp(self):
        _, self.goodfile = tempfile.mkstemp()
        with open(self.goodfile, 'w') as fh:
            fh.write("% This is a file\n%Clusters 3 cat,dog,bear\n% foobar")

        _, self.nocluster = tempfile.mkstemp()
        with open(self.nocluster, 'w') as fh:
            fh.write("% This is a file\n%Nope 3 cat,dog,bear\n% foobar")

        _, self.mismatch = tempfile.mkstemp()
        with open(self.mismatch, 'w') as fh:
            fh.write("% This is a file\n%Clusters 5 cat,dog,bear\n% foobar")

    def tearDown(self):
        os.remove(self.goodfile)
        os.remove(self.nocluster)
        os.remove(self.mismatch)

    def test_found_classes(self):
        cls = ClassFile(self.goodfile)
        cls.get_classes()
        self.assertEqual(cls.classes, ['cat', 'dog', 'bear'])
        self.assertEqual(3, cls.number)

    def test_missing_file(self):
        with self.assertRaises(IOError):
            ClassFile('nofile')

    def test_missing_classes(self):
        cls = ClassFile(self.nocluster)
        with self.assertRaises(RuntimeError):
            cls.get_classes()

    def test_wrong_number(self):
        cls = ClassFile(self.mismatch)
        with self.assertRaises(ValueError):
            cls.get_classes()


class TestClassFileBody(unittest.TestCase):
    """Test the extraction of class information from the body."""
    def setUp(self):
        self.labels = [0, 1, 1, 0, 3]
        _, self.goodfile = tempfile.mkstemp()
        with open(self.goodfile, 'w') as fh:
            fh.write("% This is a file\n%Clusters 3 cat,dog,bear\n% foobar\n")
            fh.write("0 0\n1 1\n2 1\n3 0\n4 3")

        _, self.badfile = tempfile.mkstemp()
        with open(self.badfile, 'w') as fh:
            fh.write("% This is a file\n%Clusters 3 cat,dog,bear\n% foobar\n")
            fh.write("0,1\n1 \n2 1\n3 0\n4 3")

    def tearDown(self):
        os.remove(self.goodfile)
        os.remove(self.badfile)
#        os.remove(self.mismatch)

    def test_good_load(self):
        cls = ClassFile(self.goodfile)
        cls.read()
        self.assertEqual(['cat', 'dog', 'bear'], cls.classes)
        self.assertEqual(3, cls.number)
        self.assertTrue(numpy.array_equal(self.labels, cls.labels))
        self.assertEqual(5, cls.n_samples)

    def test_good_header_loaded(self):
        cls = ClassFile(self.goodfile)
        cls.get_classes()
        cls.read()
        self.assertEqual(['cat', 'dog', 'bear'], cls.classes)
        self.assertEqual(3, cls.number)
        self.assertTrue(numpy.array_equal(self.labels, cls.labels))
        self.assertEqual(5, cls.n_samples)

    def test_missing_file(self):
        with self.assertRaises(IOError):
            ClassFile('nofile').read()

    def test_badfile(self):
        cls = ClassFile(self.badfile)
        with self.assertRaises(ValueError):
            cls.read()


class TestWordCountFile(unittest.TestCase):
    """Test the Word Count class."""

    def setUp(self):
        _, self.file = tempfile.mkstemp()

    def tearDown(self):
        os.remove(self.file)

    def test_good_case(self):
        mmf = """%%MatrixMarket matrix coordinate real general
2 3 4
1 1 1.0
1 3 5.0
2 1 2.0
2 2 2.0"""
        dense_matrix = numpy.array([[1.0, 2.0], [0.0, 2.0], [5.0, 0]])
        n_samples = 3
        n_terms = 2
        with open(self.file, 'w') as fh:
            fh.write(mmf)

        wcf = WordCountFile(self.file)
        wcf.read()
        self.assertTrue(numpy.array_equal(dense_matrix, wcf.array))
        self.assertEqual(n_samples, wcf.n_samples)
        self.assertEqual(n_terms, wcf.n_terms)

    def test_missing_file(self):
        with self.assertRaises(IOError):
            WordCountFile('nofile')

    def test_invalid(self):
        mmf = """%%MatrixMarket matrix coordinate real general
2 3 5
1 1 1.0
1 3 5.0
2 1 2.0
2 2 2.0"""
        with open(self.file, 'w') as fh:
            fh.write(mmf)
        with self.assertRaises(ValueError):
            wcf = WordCountFile(self.file)
            wcf.read()


class TestTermsFile(unittest.TestCase):
    """Test the TermsFile class."""

    def setUp(self):
        _, self.file = tempfile.mkstemp()

    def tearDown(self):
        os.remove(self.file)

    def test_good_case(self):
        expected = ['line1', 'line2', 'line3']
        n_expected = 3
        with open(self.file, 'w') as fh:
            fh.writelines("\n".join(expected))

        tf = TermsFile(self.file)
        tf.read()
        self.assertEqual(expected, tf.words)
        self.assertEqual(n_expected, tf.n_terms)

    def test_missing_file(self):
        with self.assertRaises(IOError):
            TermsFile('nofile')


if __name__ == '__main__':
    unittest.main()
