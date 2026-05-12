RANDOM_SEED = 42

# Fraction of the dataset to use for training and evaluation

# # FULL
# # ORIGINAL DATASET COUNTS:
# # Disaster	TRUE	170040	244,403     1.74%
# # 	        FALSE	74363               0.76%
# # Weather			            51,033      0.52%
# # Out Topic			          9,449,777   96.97%

# LOCAL
DATA_FRACTION = 0.1

# # CLOUD
# DATA_FRACTION = 1.0

DATA_DISASTER_FRACTION = 0.30
DATA_WEATHER_FRACTION = 0.25
DATA_OUT_TOPIC_FRACTION = 0.015


def get_data_disaster_fraction():
    return DATA_DISASTER_FRACTION * DATA_FRACTION


def get_data_weather_fraction():
    return DATA_WEATHER_FRACTION * DATA_FRACTION


def get_data_out_topic_fraction():
    return DATA_OUT_TOPIC_FRACTION * DATA_FRACTION
