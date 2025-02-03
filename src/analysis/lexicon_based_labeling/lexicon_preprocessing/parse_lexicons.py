
import csv
import json

from collections import defaultdict

from utils import ROOT



def parse_liwc(in_fpath, out_fpath):
    # category id => category name
    categories = {} 

    # word => categories
    lexicon = {}

    # '%' signals a change in the .dic file.
    # (0-1) ids, cats
    # (>1) words, cat_ids (list)
    percent_sign_count = 0

    for line in open(in_fpath):

        line = line.strip()

        if len(line) == 0:
            continue

        parts = line.split('\t')

        if parts[0] == '%':
            percent_sign_count += 1
        else:
            # if the percent sign counter is 1, parse the LIWC categories
            if percent_sign_count == 1:
                categories[parts[0]] = parts[1]
            # else, parse lexicon
            else:
                lexicon[parts[0]] = [categories[cat_id] for cat_id in parts[1:]]

    # regroup the data: category => [words]
    lexicon_lists = defaultdict(list)

    for word, cats in lexicon.items():
        for cat in cats:
            lexicon_lists[cat].append(word)

    # output to json
    json.dump(lexicon_lists, open(out_fpath, "w"), indent=2)
    
    print(f"LIWC | categories = {len(lexicon_lists)}, words = {len(lexicon)}")


def parse_empath(in_fpath, out_fpath):
    categories = defaultdict(list)
    n_terms = 0

    for line in open(in_fpath, "r"):
        cols = line.strip().split("\t")
        name = cols[0]
        terms = cols[1:]
        for term in set(terms):
            categories[name].append(term)
            n_terms += 1

    # output to json
    json.dump(categories, open(out_fpath, "w"), indent=2)

    print(f"Empath | categories = {len(categories)}, terms = {n_terms}")


def parse_nrc_affect_lexicon(in_fpath, out_fpath):
    labels = [
        "anger",
        "fear",
        "sadness",
        "joy"
    ]

    # word => {affect: score ... } (e.g. bathtub => {joy: 0.156})
    word_affect_scores = {}

    for row in csv.DictReader(open(in_fpath), dialect="excel-tab"):
        word = row["term"]
        dimension = row["AffectDimension"]
        score = float(row["score"])

        # there a term TRUE in the lexicon => skipping it
        if not word.islower():
            continue

        # sanity check
        assert dimension in labels

        w_scores = word_affect_scores.get(word, {})
        w_scores[dimension] = score
        word_affect_scores[word] = w_scores

    # output to json
    json.dump(
        {"lexicon": word_affect_scores, "labels": labels},
        open(out_fpath, "w"),
        indent=2
    )

    print(f"NRC Affect Lexicon | N = {len(word_affect_scores)}")


def parse_nrc_emotions_lexicon(in_fpath, out_fpath):
    labels = [
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "negative",
        "positive",
        "sadness",
        "surprise",
        "trust"
    ]

    # word => {emotion1: 0 / 1, emotion2: 0 / 1, ...}
    word_emotions = {}

    for i, row in enumerate(csv.reader(open(in_fpath), dialect="excel-tab")):
        if i == 0:
            continue

        word, affect, association = row

        association = int(association)

        # make sure that the word in lower case
        assert word.islower()

        # sanity check: known affect
        assert affect in labels

        w_emotions = word_emotions.get(word, {})
        w_emotions[affect] = association
        word_emotions[word] = w_emotions

    # sanity check: each word has 10 emotion labels (0 or 1)
    for word, emotions in word_emotions.items():
        assert len(emotions) == 10

    # output to json
    json.dump(
        {"lexicon": word_emotions, "labels": labels},
        open(out_fpath, "w"),
        indent=2
    )

    print(f"NRC Emotions Lexicon | N = {len(word_emotions)}")


def parse_nrc_vad_lexicon(in_fpath, out_fpath):
    labels = [
        "valence",
        "arousal",
        "dominance"
    ]

    # word => {valence: x, arousal: x, dominance: x}
    word_vad = {}

    for row in csv.DictReader(open(in_fpath), dialect="excel-tab"):
        word = row["Word"]
        valence = float(row["Valence"])
        arousal = float(row["Arousal"])
        dominance = float(row["Dominance"])

        # sanity checks
        assert word.islower()
        assert word not in word_vad

        word_vad[word] = {
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance
        }

    # output to json
    json.dump(
        {"lexicon": word_vad, "labels": labels},
        open(out_fpath, "w"),
        indent=2
    )

    print(f"NRC Word VAD | N = {len(word_vad)}")


def parse_controversy_lexicon(controversy_in, lexicons_dir):
    labels = [
        'controversial', 
        'non_controversial', 
        'medium_controversial'
    ]
    label_files = {l:open(f"{lexicons_dir}/{l}.txt", "w") for l in labels}
    n_words = {l:0 for l in labels}

    for line in open(controversy_in):
        word, label = line.strip().split("\t")
        word = word.strip().lower()
        label = label.strip().lower().replace("-", "_")

        label_files[label].write(f"{word}\n")
        n_words[label] += 1

    for f in label_files.values():
        f.close()

    print(f"Controversy lexicon | {n_words} | N = {sum(n_words.values())}")


def parse_mpqa(mpqa_in, lexicons_dir):

    word_subj_all = defaultdict(set)
    word_pol_all = defaultdict(set)

    for line in open(mpqa_in):
        parts = line.strip().split()
        word_dict = dict([part.split("=") for part in parts])

        word = word_dict["word1"]
        subj = word_dict["type"]
        pol = word_dict["priorpolarity"]

        word_subj_all[word].add(subj)
        word_pol_all[word].add(pol)
    
    # remove words with ambiguous labels (i.e., that depend on POS)
    word_subj = {w: s.pop() for w, s in word_subj_all.items() if len(s) == 1}
    word_pol = {w: p.pop() for w, p in word_pol_all.items() if len(p) == 1}

    # output subjectivity lexicons
    subj_file_map = {
        "strongsubj": open(f"{lexicons_dir}/strong_subjectives_mpqa.txt", "w"),
        "weaksubj": open(f"{lexicons_dir}/weak_subjectives_mpqa.txt", "w")
    }
    subj_counts = defaultdict(int)

    for word, subj in word_subj.items():
        subj_file_map[subj].write(f"{word}\n")
        subj_counts[subj] += 1

    for f in subj_file_map.values():
        f.close()

    print(f"Subjectivity, {dict(subj_counts)}| N = {sum(subj_counts.values())}")

    # output polarity lexicons
    pol_file_map = {
        "positive": open(f"{lexicons_dir}/pos_mpqa.txt", "w"),
        "negative": open(f"{lexicons_dir}/neg_mpqa.txt", "w"),
        "neutral": open(f"{lexicons_dir}/neu_mpqa.txt", "w")
    }
    pol_counts = defaultdict(int)

    for word, pol in word_pol.items():
        if pol not in pol_file_map:
            continue
        pol_file_map[pol].write(f"{word}\n")
        pol_counts[pol] += 1

    for f in pol_file_map.values():
        f.close()
    
    print(f"Polarity, {dict(pol_counts)}, N = {sum(pol_counts.values())}")



def main():
    
    LEXICONS_DIR = f"{ROOT}/lexical_analysis/lexicons"
    
    # Empath
    empath_in = f"{LEXICONS_DIR}/_raw/empath/empath.tsv"
    empath_out = f"{LEXICONS_DIR}/parsed/empath.json"    
    parse_empath(empath_in, empath_out)

    # LIWC (2015)
    liwc_in = f"{LEXICONS_DIR}/_raw/liwc/LIWC_2015.dic"
    liwc_out = f"{LEXICONS_DIR}/parsed/liwc.json"
    parse_liwc(liwc_in, liwc_out)

    # NRC Affect
    nrc_affect_in = f"{LEXICONS_DIR}/_raw/nrc/affect/NRC-AffectIntensity-Lexicon.txt"
    nrc_affect_out = f"{LEXICONS_DIR}/parsed/nrc_affect.json"
    parse_nrc_affect_lexicon(nrc_affect_in, nrc_affect_out)

    # NRC Emotions
    nrc_emotions_in = f"{LEXICONS_DIR}/_raw/nrc/emotions/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    nrc_emotions_out = f"{LEXICONS_DIR}/parsed/nrc_emotions.json"
    parse_nrc_emotions_lexicon(nrc_emotions_in, nrc_emotions_out)

    # NRC VAD (Valence, Arousal, and Dominance)
    nrc_vad_in = f"{LEXICONS_DIR}/_raw/nrc/vad/NRC-VAD-Lexicon.txt"
    nrc_vad_out = f"{LEXICONS_DIR}/parsed/nrc_vad.json"
    parse_nrc_vad_lexicon(nrc_vad_in, nrc_vad_out)

    # Controversy
    controversy_in = f"{LEXICONS_DIR}/_raw/controversy/controversial_words.txt"
    parse_controversy_lexicon(controversy_in, f"{LEXICONS_DIR}/parsed/")

    # MPQA: subjectivity & polarity
    mpqa_in = f"{LEXICONS_DIR}/_raw/mpqa/subjclueslen1-HLTEMNLP05.tff"
    parse_mpqa(mpqa_in, f"{LEXICONS_DIR}/parsed/")
    

    """
    Empath | categories = 194, terms = 16159
    LIWC | categories = 73, words = 6547
    NRC Affect Lexicon | N = 4191
    NRC Emotions Lexicon | N = 14182
    NRC Word VAD | N = 20007
    Controversy lexicon | {'controversial': 145, 'non_controversial': 272, 'medium_controversial': 45} | N = 462
    Subjectivity, {'weaksubj': 2139, 'strongsubj': 4695}| N = 6834
    Polarity, {'negative': 4143, 'positive': 2289, 'neutral': 417}, N = 6849
    """
    

if __name__ == "__main__":
    main()

# END
