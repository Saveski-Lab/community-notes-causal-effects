

import re
import spacy
import os

from spacy.tokens import Token, Span, Doc
from functools import reduce
from boltons.iterutils import pairwise


def load_blocklist(path):
    """Read block list, or empty set.
    """
    try:
        with open(path) as fh:
            return set(fh.read().splitlines())

    except:
        return set()


BLOCKLIST_PATH = os.path.join(os.path.dirname(__file__), 'blocklist.txt')
BLOCKLIST = load_blocklist(BLOCKLIST_PATH)


# TODO: What to do with "/"?


QUOTES = (('“', '"'), ('”', '"'), ('‘', "'"), ('’', "'"))

def standardize_quotes(text):
    """Curly -> straight.
    """
    for special, standard in QUOTES:
        text = text.replace(special, standard)

    return text


# http://jkorpela.fi/dashes.html
HYPHENS = {u'\u002D', u'\u2010', u'\u2011', u'\u2212', u'\uFE63'}

def standardize_hyphens(text):
    """Hyphen variants -> "-".
    """
    for h in HYPHENS:
        text = text.replace(h, '-')

    # Drop infix hyphens.
    text = re.sub('([^\s])-([^\s])', r'\1 \2', text)

    return text


def standardize_and(text):
    return text.replace(' & ', ' and ')


def drop_twitter_chars(text):
    return re.sub('[@#]', '', text)


STANDARDIZERS = (
    standardize_quotes,
    standardize_hyphens,
    standardize_and,
    drop_twitter_chars,
)

def standardize_text(text):
    return reduce(lambda t, func: func(t), STANDARDIZERS, text)


# Headlines can contain letters, digits, ".,;", "?!", "$%,"" and spaces.
BREAK_CHAR_PATTERN = '[^a-z0-9\s\.,;\?!\'"\$%]'

# Regular tokens that constitute a break.
ALPHA_BREAK_TOKENS = {'via'}

def is_break_token(token):
    """Does the token constitute a "sentence" break?
    """
    text = token.text.lower()

    has_break_char = bool(re.search(BREAK_CHAR_PATTERN, text))
    is_alpha_break_token = text in ALPHA_BREAK_TOKENS

    return has_break_char or is_alpha_break_token


# For the classifier, drop everything except letters, numbers, and $.
CLF_REMOVED_CHAR_PATTERN = '[^a-z0-9\$%]'

def token_clf_text(token):
    """Drop everything but letters, numbers, and currency ($.,)
    """
    # Drop everything but letters, numbers, $.
    text = re.sub(CLF_REMOVED_CHAR_PATTERN, '', token.text.lower())

    # Digits -> '#'.
    text = re.sub('[0-9]+', '#', text)

    return text


def span_clf_text(span):
    return ' '.join(t._.clf_text for t in span if t._.clf_text).strip()


def break_idxs(doc):
    """Locations of break tokens, with left/right bookends.
    """
    return [-1, *[t.i for t in doc if t._.is_break_token], len(doc)]


def spans(doc):
    """Pull apart separator-delimited spans.
    """
    return [doc[i1+1:i2] for i1, i2 in pairwise(doc._.break_idxs)]


def span_clf_texts(doc):
    """Join spans -> clf strings, drop empty strings.
    """
    texts = [span._.clf_text for span in doc._.spans]
    return tuple(filter(bool, texts))


def longest_unbroken_span(doc):
    """Find the longest span of tokens without a break token.
    """
    return sorted(doc._.spans, key=lambda s: len(s), reverse=True)[0]


def clf_tokens(doc):
    """Get filtered tokens for classifier.
    """
    return [
        token
        for span in doc._.spans
        if span._.clf_text and span._.clf_text not in BLOCKLIST
        for token in span
        if token._.clf_text
    ]


def clf_token_texts(doc):
    return [t._.clf_text for t in doc._.clf_tokens]


Token.set_extension('is_break_token', getter=is_break_token)
Token.set_extension('clf_text', getter=token_clf_text)

Span.set_extension('clf_text', getter=span_clf_text)

Doc.set_extension('break_idxs', getter=break_idxs)
Doc.set_extension('spans', getter=spans)
Doc.set_extension('span_clf_texts', getter=span_clf_texts)
Doc.set_extension('longest_unbroken_span', getter=longest_unbroken_span)
Doc.set_extension('clf_tokens', getter=clf_tokens)
Doc.set_extension('clf_token_texts', getter=clf_token_texts)

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])


def parse_headline(text):
    return nlp(standardize_text(text))
