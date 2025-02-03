
import gzip
import csv
import ujson as json

from tqdm import tqdm
from utils import read_tweets
from utils import ROOT


def main():
    
    INPUT = f"{ROOT}/firehose-tweets-2017-2020/news_tweets_outcomes.json.gz"
    OUTPUT = f"{ROOT}/lexical_analysis/liwc/in_tweet_body.csv"
    USE_TOKENS = False

    # CSV output setup
    fieldnames = ["tweet_id", "text"]
    csv_writer = csv.DictWriter(
        open(OUTPUT, "w"),
        fieldnames=fieldnames,
        quoting=csv.QUOTE_ALL
    )
    csv_writer.writeheader()

    # main loop
    for tweet in tqdm(read_tweets(INPUT, MAX_RECS=None)):

        tweet_id = tweet["tweet_id"]
        text = tweet["body_processed"]
        if USE_TOKENS:
            text = " ".join(tweet["body_processed_tokens"])

        csv_writer.writerow({
            "tweet_id": tweet_id,
            "text": text
        })

    print("Done!")


if __name__ == "__main__":
    main()

# END
