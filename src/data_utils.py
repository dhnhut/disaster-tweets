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
DATA_DISASTER_FRACTION = 0.25  # => rerun the 1_0_disaster_datasets
DATA_WEATHER_FRACTION = 1.0
DATA_OUT_TOPIC_FRACTION = 0.0031


def get_data_fraction():
    return DATA_FRACTION


def get_data_disaster_fraction():
    return DATA_DISASTER_FRACTION * DATA_FRACTION


def get_data_weather_fraction():
    return DATA_WEATHER_FRACTION * DATA_FRACTION


def get_data_out_topic_fraction():
    return DATA_OUT_TOPIC_FRACTION * DATA_FRACTION


def get_data_path(type: str = ""):
    path = Path(f"../data/{type}")
    if type == "disaster":
        path = Path(f"../data/splited/{get_data_disaster_fraction()}")
    elif type == "weather":
        path = Path(f"../data/splited/{get_data_weather_fraction()}")
    elif type == "out_topic":
        path = Path(f"../data/splited/{get_data_out_topic_fraction()}")

    return Path(f"../data/{type}")


def get_data_set(set_name, label="informative", frac: float = get_data_fraction()):
    path = get_data_path("splited") / str(frac)
    return path / f"{label}_{set_name}.csv"


def load_datasets(label="informative", frac: float = get_data_fraction()):
    df_train = pd.read_csv(get_data_set("train", label=label, frac=frac))
    df_val = pd.read_csv(get_data_set("validation", label=label, frac=frac))
    df_test = pd.read_csv(get_data_set("test", label=label, frac=frac))
    return df_train, df_val, df_test
