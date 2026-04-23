from emoji import demojize
from nltk.tokenize import TweetTokenizer
from src.normalizer.emoticons import EMOTICON_MAP, EMOTICON_SORTED_KEYS
from src.normalizer.abbreviations import ABBREVIATION_MAP, ABBREVIATION_SORTED_KEYS


# https://www.nltk.org/api/nltk.tokenize.casual.html
tokenizer = TweetTokenizer(
    preserve_case=True,
    reduce_len=True,
    strip_handles=False,
    match_phone_numbers=False,
)


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    return token


def normalizeToken_BERTweet(tweet):
    # BERTweet
    tweet = (
        tweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
        .replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
        .replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    for emoticon in EMOTICON_SORTED_KEYS:
        tweet = tweet.replace(emoticon, EMOTICON_MAP[emoticon])
    for abbr in ABBREVIATION_SORTED_KEYS:
        tweet = tweet.replace(abbr, ABBREVIATION_MAP[abbr])

    tweet = demojize(tweet)

    return tweet


def normalize_tweet(tweet, is_BERTweet=False):
    tweet = tweet.replace("’", "'").replace("…", "...")
    tokens = tokenizer.tokenize(tweet)
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    if is_BERTweet:
        normTweet = normalizeToken_BERTweet(normTweet)

    return " ".join(normTweet.split())


if __name__ == "__main__":
    text = "Bizarre side effect of solar panels causes rainstorms in the driest place on Earthhhhhhhh!!!! @postandcourier 😢!!!!!!!! https://www.uniladtech.com/science/news/bizarre-side-effect-solar-panels-rainstorms-driest-place-540247-20260417?fbclid=IwY2xjawRQad5leHRuA2FlbQIxMABzcnRjBmFwcF9pZBAyMjIwMzkxNzg4MjAwODkyAAEeus1zhw0Y5Q1kjMUE8uKoAw3aNQn4hIpWFUrFtsdxE3BC3dnsQe7g4vTynLY_aem_TwSO6h1VOuEZhpJ9Ihv9_g…"
    print(
        normalize_tweet(
            text,
        )
    )
    print(
        normalize_tweet(
            text,
            is_BERTweet=True,
        )
    )
