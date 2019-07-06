import os
import unittest
import tempfile
import collections
import logging

logger = logging.getLogger()

from articlass.text import load_words, html2text, count_terms


class TestLoadWords(unittest.TestCase):
    """Test the load of a list of words."""
    def setUp(self):
        _, self.file = tempfile.mkstemp()

    def tearDown(self):
        os.remove(self.file)

    def test_good_load(self):
        expected = ['line1', 'line2', 'line3']
        with open(self.file, 'w') as fh:
            fh.writelines("\n".join(expected))
        result = load_words(self.file)
        self.assertEqual(expected, result)

    def test_whitespace_strip(self):
        expected = ['line1', 'line2', 'line3']
        write = ['line1  ', 'line2\t', '  line3']
        with open(self.file, 'w') as fh:
            fh.writelines("\n".join(write))
        result = load_words(self.file)
        self.assertEqual(expected, result)

    def test_oneline(self):
        expected = ['line1line2line3']
        with open(self.file, 'w') as fh:
            fh.write(expected[0])
        result = load_words(self.file)
        self.assertEqual(expected, result)

    def test_missing_file(self):
        with self.assertRaises(IOError):
            load_words('nofile')

    def test_empty_file(self):
        with open(self.file, 'w') as fh:
            fh.write('')
        with self.assertRaises(EOFError):
            load_words(self.file)


class TestHtml2Text(unittest.TestCase):
    """Test the conversion from html to plain text."""

    def test_good_case(self):
        html = ("<body>Filtered text<p>Line 1</p><p>Line 2</p>"
                "<p class=skip>Line X</p>")
        expected = "Line 1 Line 2"
        result = html2text(html)
        self.assertEqual(expected, result)

    def test_no_text(self):
        html = " "
        with self.assertRaises(ValueError):
            html2text(html)

    def test_no_valid_text(self):
        html = "<p class=skip>Should not count</p> "
        with self.assertRaises(ValueError):
            html2text(html)


class TestCountTerms(unittest.TestCase):
    """Test the conversion from html to plain text."""

    def test_good_case(self):
        counts = collections.Counter(
            ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd'])
        terms = ['a', 'b', 'c', 'd']
        expected = [5, 3, 0, 0]
        result = count_terms(counts, terms)
        self.assertEqual(expected, result)

    def test_no_matches(self):
        counts = collections.Counter(
            ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd'])
        terms = ['e', 'f']
        expected = [0, 0]
        result = count_terms(counts, terms)
        self.assertEqual(expected, result)

    def test_incorrect_type(self):
        counts = {'a': 5, 'b': 3, 'c': 2, 'd': 1}
        terms = ['a', 'b', 'c', 'd']
        with self.assertRaises(TypeError):
            count_terms(counts, terms)


if __name__ == '__main__':
    unittest.main()
