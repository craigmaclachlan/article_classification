"""Module containing text processing functions."""

import bs4
import collections


def load_words(path):
    """
    Load a list of words from a text file.

    Args:
        path - path to a file containing line separated words
    Returns
        List of strings

    """
    with open(path, 'r') as fh:
        words = [w.strip() for w in fh.readlines()]
        if len(words) == 0:
            raise EOFError("No text found in file")
        return words


def html2text(html):
    """
    Convert some tagged HTML into plain text.

    Args:
        html - a string containing HTML.
    Returns:
        A string containing all of the text.

    This has been designed to work, with the current BBC website (June 2019).
    We assume that the text is contained in the 'p' tags and that any tags
    with attributes are special in some way, and consequently will be filtered
    out.

    """
    soup = bs4.BeautifulSoup(html, "html.parser")
    text_list = [tag.text for tag in soup.findAll('p') if tag.attrs == {}]
    text = " ".join(text_list).strip()
    if text == '':
        raise ValueError("No text found.")
    return " ".join(text_list)


def count_terms(counts, terms):
    """
    For each word in a given list, return the number of times it
    appears in count.

    The original training data had low frequency words filtered. Any
    words with counts less than 3 were not included. We need to do the
    same here because the neural net hasn't learned what to do when
    there are counts of 1 or 2 - it may give unexpected results!

    Args:
        counts (collections.Counter) - dictionary of words and their counts.
        terms - predefined list of words that need counts.

     Returns:
         list of integers, the counts of each term

    """
    if not isinstance(counts, collections.Counter):
        raise TypeError("Counts must be of type collections.Counter.")
    term_counts = []
    for w in terms:
        # The Counter class is helpful here; it will return zero if we try
        # to index a key not in the dictionary.
        value = counts[w]
        term_counts.append(
            value if value >= 3 else 0
        )
    return term_counts
