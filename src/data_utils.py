from pathlib import Path

import pandas as pd

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
DATA_FRACTION = 0.1  # => rerun the 1_0_disaster_datasets
# # CLOUD
# DATA_FRACTION = 1.0

DATA_DISASTER_FRACTION = 0.5  # => rerun the 1_0_disaster_datasets
DATA_WEATHER_FRACTION = 1.0
DATA_OUT_TOPIC_FRACTION = 0.0031

EXPERIMENT_RATIOS = [
    # weather, out-topic
    [0, 0],
    [0.5, 0],
    [1, 0.0196],
    [1, 0.0334],
    [1, 0.0610],
    [1, 0.1301],
    [1, 0.2682],
]

INFORMATIVE_FILE = "df_disaster_informative_knearest_0.75_100.csv"
WEATHER_FILE = "df_weather_radius_0.75.csv"
OUT_TOPIC_FILE = "df_out_topic_knearest_0.6_top_100.csv"


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
        return path / f"../data/splited/{get_data_disaster_fraction()}"
    elif type == "weather":
        return path / f"../data/splited/{get_data_weather_fraction()}"
    elif type == "out_topic":
        return path / f"../data/splited/{get_data_out_topic_fraction()}"
    else:
        return path


def get_data_set(
    set_name,
    label="informative",
    weather_ratio=None,
    out_topic_ratio=None,
):
    path = get_data_path("splited") / get_experiment_ratios_path(
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
