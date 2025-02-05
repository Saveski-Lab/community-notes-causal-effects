import json

from src.utils import shared_data_root

def read_lexicon(fpath):
    lexicon = set([])

    for line in open(fpath):
        line = line.strip()
        
        if line.startswith("#") or line.startswith(";"):
            continue

        if len(line) == 0 or len(line.split()) != 1:
            continue
        
        lexicon.add(line)
        
    return lexicon


LEXICONS_INFO = [
    ("assertives", "assertives_hooper1975.txt"),
    ("entailed", "entailed_berant2012.txt"),
    ("entailed_arg", "entailed_arg_berant2012.txt"),
    ("entailing", "entailing_berant2012.txt"),
    ("entailing_arg", "entailing_arg_berant2012.txt"),
    ("factives", "factives_hooper1975.txt"),
    ("hedges", "hedges_hyland2005.txt"),
    ("implicatives", "implicatives_karttunen1971.txt"),
    ("pos_liu", "positive_liu2005.txt"),
    ("neg_liu", "negative_liu2005.txt"),
    ("pos_mpqa", "pos_mpqa.txt"),
    ("neu_mpqa", "neu_mpqa.txt"),
    ("neg_mpqa", "neg_mpqa.txt"),
    ("subj_strong_riloff", "strong_subjectives_riloff2003.txt"),
    ("subj_weak_riloff", "weak_subjectives_riloff2003.txt"),
    ("subj_strong_mpqa", "strong_subjectives_mpqa.txt"),
    ("subj_weak_mpqa", "weak_subjectives_mpqa.txt"),
    ("controversial", "controversial.txt"),
    ("non_controversial", "non_controversial.txt"),
    ("medium_controversial", "medium_controversial.txt")
]

def load_all_lexicons():

    #
    # read (non-valued) lexicons
    #
    lexicons = []
    for l_name, l_fname in LEXICONS_INFO:
        l_fpath = shared_data_root / "lexicons" / "lexicons" / l_fname
        lexicon = read_lexicon(l_fpath)
        lexicons.append({
            "name": l_name,
            "lexicon": lexicon
        })

    #
    # read valued lexicons
    #
    valued_lexicons_info = [
        ("nrc_affect", "nrc_affect.json"),
        ("nrc_emotions", "nrc_emotions.json"),
        ("nrc_vad", "nrc_vad.json")
    ]

    valued_lexicons = []

    for l_name, l_fname in valued_lexicons_info:
        l_fpath = shared_data_root / "lexicons" / "lexicons" / l_fname
        lexicon = json.load(open(l_fpath))
        valued_lexicons.append({
            "name": l_name,
            "labels": lexicon["labels"],
            "lexicon": lexicon["lexicon"]
        })


    #
    # compile CSV fields
    #
    fields = []

    fields += [lexicon["name"] for lexicon in lexicons]

    for valued_lexicon in valued_lexicons:
        l_name = valued_lexicon["name"]
        l_labels = valued_lexicon["labels"]
        fields += [f"{l_name}_{label}_sum" for label in l_labels]
        fields += [f"{l_name}_count"]

    fields += ["token_count"]

    return lexicons, valued_lexicons, fields


def process_tokens(tokens, lexicons, valued_lexicons, fields):
    tweet_labels = {f: 0 for f in fields}

    # Filter out empty strings
    if len(tokens) == 0:
        return tweet_labels

    # lexicons
    for token in tokens:
        for lexicon in lexicons:
            l_name = lexicon["name"]
            l_lexicon = lexicon["lexicon"]

            if token not in l_lexicon:
                continue

            tweet_labels[l_name] += 1

    # Normalize by token count
    for lexicon in lexicons:
        l_name = lexicon["name"]
        tweet_labels[l_name] /= len(tokens)

    # valued lexicons
    for token in tokens:
        for valued_lexicon in valued_lexicons:
            l_name = valued_lexicon["name"]
            l_lexicon = valued_lexicon["lexicon"]

            if token not in l_lexicon:
                continue

            for label, value in l_lexicon[token].items():
                tweet_labels[f"{l_name}_{label}_sum"] += value

            tweet_labels[f"{l_name}_count"] += 1

    # Normalize by number of tokens that were matched:
    for valued_lexicon in valued_lexicons:
        l_name = valued_lexicon["name"]
        for label in valued_lexicon["labels"]:
            if tweet_labels[f"{l_name}_count"] > 0:
                tweet_labels[f"{l_name}_{label}_sum"] /= tweet_labels[f"{l_name}_count"]

    # Add token count as field
    tweet_labels["token_count"] = len(tokens)

    return tweet_labels
