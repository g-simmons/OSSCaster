from pathlib import Path
import os

ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models/"
REFORMAT_DATA_DIR = ROOT / "Sustainability_Analysis/Reformat_data/"
# MODELS_DIR = ROOT / "Sustainability_Analysis/4_models/"

RANDOM_STATE = 42

MAX_N_TIMESTEPS = 20  # TODO: this should be set to the longest model, probably same as the longest sequence in the training data
N_TIMESTEPS = 8

DATA_COLUMNS = [
    "active_devs",
    "num_commits",
    "num_files",
    "num_emails",
    "c_percentage",
    "e_percentage",
    "inactive_c",
    "inactive_e",
    "c_nodes",
    "c_edges",
    "c_c_coef",
    "c_mean_degree",
    "c_long_tail",
    "e_nodes",
    "e_edges",
    "e_c_coef",
    "e_mean_degree",
    "e_long_tail",
]
