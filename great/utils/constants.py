import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(FILE_PATH, "../../")  # path of the GREAT folder

# location where LKH code is located, needs to be adjusted depending on the machine
# if needed, download code from here: http://webhotel4.ruc.dk/~keld/research/LKH/
# the LKH code in this location has been adjusted to only generate the candidate sets and not compute actual TSP tours (--> faster)
LKH_CANDIDATE_PATH = os.path.join(BASE_PATH, "../candidate_wrapper")

# location where LKH code is located, needs to be adjusted depending on the machine
# if needed, download code from here: http://webhotel4.ruc.dk/~keld/research/LKH/
LKH_SOLVER_PATH = os.path.join(BASE_PATH, "../lkh_wrapper")


# data path (where data shall be stored and loaded from)
DATA_PATH = os.path.join(BASE_PATH, "great/dataset/data")
TEST_DATA_PATH = os.path.join(DATA_PATH, "test_data")
TEST_DATA_LOAD_PATH = os.path.join(BASE_PATH, "final_models/TSP_RL")


# model path (where shall trained models be saved?)
MODEL_PATH = os.path.join(BASE_PATH, "models")

# final model path (there the trained models are saved that we ship with the repository)
FINAL_MODEL_PATH = os.path.join(BASE_PATH, "final_models")

# config paths for the different "main" scripts using hydra
BENCHMARKING_RL_CONFIGS_PATH = os.path.join(BASE_PATH, "final_models/configs")
BENCHMARKING_SL_CONFIGS_PATH = os.path.join(BASE_PATH, "final_models/TSP_SL_TOUR")
TRAIN_CONFIG_PATH = os.path.join(BASE_PATH, "config_files")


# figure path to save figures
FIGURE_PATH = os.path.join(BASE_PATH, "figures")
