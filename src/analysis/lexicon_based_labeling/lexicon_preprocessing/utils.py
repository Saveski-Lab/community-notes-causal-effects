
import gzip
import ujson as json


# ROOT = "/Users/msaveski/code/bridge/data"
ROOT = "/media/lipicaner/bridge/data"

TWEET_LANG = "en"
MIN_N_CHARS = 5
MIN_N_TOKENS = 3
MAX_TIME = "2020-02-23T00:00:00.000Z"


def read_tweets(tweets_fpath, MAX_RECS=1000):

    fin = gzip.open(tweets_fpath)
    n_recs = 0

    for line in fin:

        if MAX_RECS is not None and n_recs >= MAX_RECS:
            break

        x = json.loads(line)

        # english only
        if x["tweet_lang"] != TWEET_LANG:
            continue

        # min 5 characters
        if len(x["body_processed"]) < MIN_N_CHARS:
            continue

        # min 3 tokens
        if len(x["body_processed_tokens"]) < MIN_N_TOKENS:
            continue

        # remove the tweest from the last week
        if x["time"] > MAX_TIME:
            continue
        
        # emit record
        n_recs += 1
        yield x

