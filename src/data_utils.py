from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

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


def plot_fine_tune_history(train_loss_history, val_loss_history, val_acc_history):
    if len(train_loss_history) == 0 or len(val_loss_history) == 0:
        print("No training history found. Run the fine-tuning cell first.")
    else:
        epochs = range(1, len(train_loss_history) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(epochs, train_loss_history, marker="o", label="Train Loss")
        axes[0].plot(epochs, val_loss_history, marker="o", label="Val Loss")
        axes[0].set_title("Loss Curve")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(
            epochs, val_acc_history, marker="o", color="tab:green", label="Val Accuracy"
        )
        axes[1].set_title("Validation Accuracy Curve")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.show()
