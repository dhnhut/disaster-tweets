ABBREVIATION_MAP = {
    # Laughing
    "lol": "laughing out loud",
    "lmao": "laughing my ass off",
    "lmfao": "laughing my fucking ass off",
    "rofl": "rolling on the floor laughing",
    # Opinions & Reactions
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "imao": "in my arrogant opinion",
    "omg": "oh my god",
    "smh": "shaking my head",
    "idk": "I don't know",
    "idc": "I don't care",
    "tbh": "to be honest",
    "ngl": "not gonna lie",
    "fr": "for real",
    # Greetings & Goodbyes
    "brb": "be right back",
    "gtg": "got to go",
    "g2g": "got to go",
    "ttyl": "talk to you later",
    "cya": "see ya",
    # Information & Communication
    "btw": "by the way",
    "fyi": "for your information",
    "afaik": "as far as I know",
    "iirc": "if I recall correctly",
    "tldr": "too long; didn't read",
    "tl;dr": "too long; didn't read",
    "dm": "direct message",
    # "pm": "private message"
}

ABBREVIATION_SORTED_KEYS = sorted(
    ABBREVIATION_MAP.keys(),
    key=len,
    reverse=True,
)
