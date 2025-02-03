
import csv
import json

from collections import defaultdict

from utils import ROOT


def read_txt_lexicon(fpath):
    lexicon = set([])

    for line in open(fpath):
        line = line.strip()
        
        if line.startswith("#") or line.startswith(";"):
            continue

        if len(line) == 0 or len(line.split()) != 1:
            continue
        
        lexicon.add(line)
        
    return lexicon


def main():

    # paths
    lexicons_dir = f"{ROOT}/lexical_analysis/lexicons/parsed"
    out_csv_fpath = f"{ROOT}/lexical_analysis/lexicons/all_lexicons.csv"

    lexicon_rows = []

    #
    # Text files with lists of words
    #
    txt_files = [
        ("hooper1975", "assertives", "assertives_hooper1975.txt"),
        ("hooper1975", "factives", "factives_hooper1975.txt"),
        #
        ("mejova2014", "non_controversial", "non_controversial.txt"),
        ("mejova2014", "medium_controversial", "medium_controversial.txt"),
        ("mejova2014", "controversial", "controversial.txt"),
        #
        ("berant2012", "entailed_arg", "entailed_arg_berant2012.txt"),
        ("berant2012", "entailed", "entailed_berant2012.txt"),
        ("berant2012", "entailing_arg", "entailing_arg_berant2012.txt"),
        ("berant2012", "entailing", "entailing_berant2012.txt"),
        #
        ("hyland2005", "hedges", "hedges_hyland2005.txt"),
        #
        ("karttunen1971", "implicatives", "implicatives_karttunen1971.txt"),
        #
        ("wikipedia", "neutral_pov", "npov_lexicon.txt"),
        #
        ("liu2005", "negative", "negative_liu2005.txt"),
        ("liu2005", "positive", "positive_liu2005.txt"),
        #
        ("recasens2013", "report_verbs", "report_verbs.txt"), 
        #
        ("mpqa", "neg", "neg_mpqa.txt"),
        ("mpqa", "neu", "neu_mpqa.txt"),
        ("mpqa", "pos", "pos_mpqa.txt"),
        ("mpqa", "weak_subjectives", "weak_subjectives_mpqa.txt"),
        ("mpqa", "strong_subjectives", "strong_subjectives_mpqa.txt"),
        #
        ("riloff2003", "weak_subjectives", "weak_subjectives_riloff2003.txt"),
        ("riloff2003", "strong_subjectives", "strong_subjectives_riloff2003.txt")
    ]

    for src, category, fname in txt_files:
        words = read_txt_lexicon(f"{lexicons_dir}/{fname}")
        for word in words:
            lexicon_rows.append({
                "source": src,
                "category": category,
                "word": word
            })


    # LIWC
    liwc = json.load(open(f"{lexicons_dir}/liwc.json"))

    for category, words in liwc.items():
        for word in words:
            lexicon_rows.append({
                "source": "liwc",
                "category": category,
                "word": word
            })
    
    # Empath
    empath = json.load(open(f"{lexicons_dir}/empath.json"))

    for category, words in empath.items():
        for word in words:
            lexicon_rows.append({
                "source": "empath",
                "category": category,
                "word": word
            })

    # NRC Emotions
    nrc = json.load(open(f"{lexicons_dir}/nrc_emotions.json"))
    
    nrc_category_words = defaultdict(list)

    for word, labels in nrc["lexicon"].items():
        for label, val in labels.items():
            if int(val) == 1:
                nrc_category_words[label].append(word)

    for category, words in nrc_category_words.items():
        for word in words:
            lexicon_rows.append({
                "source": "nrc_emotions",
                "category": category,
                "word": word
            })
    
    #
    # summary
    #
    n_sources = len(set([i["source"] for i in lexicon_rows]))
    n_categories = len(set([i["category"] for i in lexicon_rows]))
    print(f"|sources| = {n_sources}")
    print(f"|categories| = {n_categories}")
    print(f"N = {len(lexicon_rows)}")
            
    #
    # output to CSV
    #
    with open(out_csv_fpath, "w") as fout:
        names = list(lexicon_rows[0].keys())
        writer = csv.DictWriter(fout, fieldnames=names, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(lexicon_rows)


if __name__ == "__main__":
    main()

# END