import argparse
from . import config as cfg
import json
import os
from . import run_all

parser = argparse.ArgumentParser()
parser.add_argument("--workdir", dest="workdir", help="The working directory.")
args = parser.parse_args()
workdir = args.workdir
if workdir:
    os.chdir(workdir)
else:
    workdir = os.getcwd()
print(f"Working directory: {os.getcwd()}")

config_path = os.path.join(workdir, "config.json")
if os.path.isfile(config_path):
    settings = json.load(open(config_path))
    cfg.BASELINE_FILENAME_TEMPLATE = settings["BASELINE_FILENAME_TEMPLATE"]
    cfg.BASELINE_URL = settings["BASELINE_URL"]
    cfg.JOURNAL_MEDLINE_FILENAME = settings["JOURNAL_MEDLINE_FILENAME"]
    cfg.MAIN_HEADING_DATA_FILENAME = settings["MAIN_HEADING_DATA_FILENAME"]
    cfg.NUM_BASELINE_FILES = settings["NUM_BASELINE_FILES"]
    cfg.SUBHEADING_DATA_FILENAME = settings["SUBHEADING_DATA_FILENAME"]
    cfg.TEST_SET_START_YEAR = settings["TEST_SET_START_YEAR"]
    cfg.TRAIN_SET_START_YEAR = settings["TRAIN_SET_START_YEAR"]
    cfg.TRAIN_END_TO_END_MODEL = settings["TRAIN_END_TO_END_MODEL"]
    cfg.SENTENCEPIECE_MODEL_START_YEAR = settings["SENTENCEPIECE_MODEL_START_YEAR"]
    cfg.SMALL_TRAIN_SET_START_YEAR = settings["SMALL_TRAIN_SET_START_YEAR"]
    cfg.SUBHEADING_MODEL_DROPOUT_RATE = settings["SUBHEADING_MODEL_DROPOUT_RATE"]
    cfg.USE_TEST_SET_PMIDS = settings["USE_TEST_SET_PMIDS"]
    cfg.TEST_SET_PMIDS_FILENAME = settings["TEST_SET_PMIDS_FILENAME"]

run_all.run(workdir)