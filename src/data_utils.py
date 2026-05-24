import csv
from pathlib import Path

import pandas as pd
from . import setup

# #####Fraction of the dataset to use for training and evaluation
# => rerun the 1_1_extended_datasets and 1_2_split_informative_datasets

# # FULL
# # ORIGINAL DATASET COUNTS:
# # Disaster	TRUE	170040	244,403     1.74%
# # 	        FALSE	74363               0.76%
# # Weather			            51,033      0.52%
# # Out Topic			          9,449,777   96.97%

# LOCAL
# small fraction for quick testing
# DATA_FRACTION = 0.1  # => rerun the 1_0_disaster_datasets
# # CLOUD
DATA_FRACTION = 1.0

DATA_DISASTER_FRACTION = 0.5  # => rerun the 1_0_disaster_datasets
DATA_WEATHER_FRACTION = 1.0
DATA_OUT_TOPIC_FRACTION = 0.0031

EXPERIMENT_RATIOS = [
    # # weather, out-topic
    [0, 0],  # no noise
    [1, 0],  # weather noise only
    [1, 0.085494],  # x2 noise
    [1, 0.135494],  # x3 noise,
    [1, 0.235494],  # x5 noise,
    [1, 0.485494],  # x10 noise,
    [1, 0.735494],  # x15 noise,
    [1, 0.985494],  # x20 noise,
]


DISASTER_FILE = "disaster_knearest_0.75_100.csv"
# DISASTER_FILE = "disaster_knearest_0.8_100.csv"
# INFORMATIVE_FILE = "df_disaster_informative_knearest_0.75_100.csv"
WEATHER_FILE = "weather_radius_0.75.csv"
OUT_TOPIC_FILE = "out_topic_knearest_0.6_top_100.csv"


def get_data_fraction():
    return DATA_FRACTION


def get_data_disaster_fraction():
    return DATA_DISASTER_FRACTION * DATA_FRACTION


def get_data_weather_fraction():
    return DATA_WEATHER_FRACTION * DATA_FRACTION


def get_data_out_topic_fraction():
    return DATA_OUT_TOPIC_FRACTION * DATA_FRACTION


def get_experiment_ratios_path(weather_ratio, out_topic_ratio):
    return Path(f"w{weather_ratio}_o{out_topic_ratio}") / str(get_data_fraction())


def get_data_path(type: str = ""):
    path = Path(f"../data/{type}")
    if type == "disaster":
        return path / f"../data/splitted/{get_data_disaster_fraction()}"
    elif type == "weather":
        return path / f"../data/splitted/{get_data_weather_fraction()}"
    elif type == "out_topic":
        return path / f"../data/splitted/{get_data_out_topic_fraction()}"
    else:
        return path


def get_data_set(
    set_name,
    label="informative",
    weather_ratio=None,
    out_topic_ratio=None,
):
    path = get_data_path("splitted") / get_experiment_ratios_path(
        weather_ratio, out_topic_ratio
    )
    return path / f"{label}_{set_name}.csv"


def load_datasets(label="informative", weather_ratio=None, out_topic_ratio=None):
    df_train = pd.read_csv(
        get_data_set(
            "train",
            label=label,
            weather_ratio=weather_ratio,
            out_topic_ratio=out_topic_ratio,
        )
    )
    df_val = pd.read_csv(
        get_data_set(
            "validation",
            label=label,
            weather_ratio=weather_ratio,
            out_topic_ratio=out_topic_ratio,
        )
    )
    df_test = pd.read_csv(
        get_data_set(
            "test",
            label=label,
            weather_ratio=weather_ratio,
            out_topic_ratio=out_topic_ratio,
        )
    )
    return df_train, df_val, df_test


def load_BERT_sets(weather_ratio=None, out_topic_ratio=None):
    path = Path(
        f"../data/splitted/BERT/{get_experiment_ratios_path(weather_ratio, out_topic_ratio)}"
    )
    df_train = pd.read_csv(path / f"train.csv")
    df_val = pd.read_csv(path / f"validation.csv")
    df_test = pd.read_csv(path / f"test.csv")
    return df_train, df_val, df_test


def split_fraction(
    df_disaster,
    df_weather,
    df_out_topic,
    fraction=DATA_FRACTION,
    out_topic_times=20,
    random_state=setup.RANDOM_SEED,
    file_path: Path = None,
):
    df = df_disaster.sample(frac=fraction, random_state=random_state)
    df = pd.concat(
        [df, df_weather.sample(frac=fraction, random_state=random_state)],
        ignore_index=True,
    )
    n_out_topic = int(len(df[df["informative"] == True]) * out_topic_times)
    df = pd.concat(
        [df, df_out_topic.sample(n=n_out_topic, random_state=random_state)],
        ignore_index=True,
    )

    print(f"Fraction set size: {len(df)}")
    print(
        f"Disaster informative samples: {df[df['subset'] == 'disaster'].groupby('informative').size()}"
    )
    print(
        f"Humanitarian samples: {df[(df['subset'] == 'disaster') & (df['humanitarian_label'].notnull())].groupby(['informative']).size()}"
    )
    print(
        f"Humanitarian sub-label samples: {df[df['subset'] == 'disaster'].groupby(['humanitarian_label', 'informative']).size()}"
    )
    print(f"Fraction set weather samples: {len(df[df['subset'] == 'weather'])}")
    print(f"Fraction set out-topic samples: {len(df[df['subset'] == 'out_topic'])}")

    display(df.head())

    # Remove the samples from the original datasets
    df_uids = set(df["uid"])
    df_disaster = df_disaster[~df_disaster["uid"].isin(df_uids)]
    df_weather = df_weather[~df_weather["uid"].isin(df_uids)]
    df_out_topic = df_out_topic[~df_out_topic["uid"].isin(df_uids)]

    df.to_csv(
        file_path,
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )
    return df, df_disaster, df_weather, df_out_topic
