import textstat as ts


def measure_readability(text):
    readability = {
        "sentence_count": ts.sentence_count(text),
        "word_count": ts.lexicon_count(text, removepunct=True),
        "syllable_count": ts.syllable_count(text, lang="en_US"),
        "flesch_reading_ease": ts.flesch_reading_ease(text),
        "smog_index": ts.smog_index(text),
        "flesch_kincaid_grade": ts.flesch_kincaid_grade(text),
        "coleman_liau_index": ts.coleman_liau_index(text),
        "automated_readability_index": ts.automated_readability_index(text),
        "dale_chall_readability_score": ts.dale_chall_readability_score(text),
        "difficult_words": ts.difficult_words(text),
        "linsear_write_formula": ts.linsear_write_formula(text),
        "gunning_fog": ts.gunning_fog(text),
        "text_standard": ts.text_standard(text, float_output=True),
    }

    return readability

