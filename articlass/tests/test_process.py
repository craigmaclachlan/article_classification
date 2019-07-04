import os
import unittest
import tempfile
import numpy
import logging
#import logging.config

#import articlass
#logging.config.dictConfig(articlass.logging_config)
#logger = logging.getLogger()

from articlass.process import ClassFile, WordCountFile, TermsFile

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

#        _, self.mismatch = tempfile.mkstemp()
#        with open(self.mismatch, 'w') as fh:
#            fh.write("% This is a file\n%Clusters 5 cat,dog,bear\n% foobar")

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

    def test_missing_file(self):
        with self.assertRaises(IOError):
            ClassFile('nofile').read()

    def test_badfile(self):
        cls = ClassFile(self.badfile)
        with self.assertRaises(ValueError):
            cls.read()

#    def test_wrong_number(self):
#        cls = ClassFile(self.mismatch)
#        with self.assertRaises(ValueError):
#            cls.get_classes()


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