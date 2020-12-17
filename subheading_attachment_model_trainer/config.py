# User settings 2020 baseline
# BASELINE_FILENAME_TEMPLATE = "pubmed20n{0:04d}.xml.gz"
# BASELINE_URL = "ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline"
# JOURNAL_MEDLINE_FILENAME = "J_Medline_31_Dec_19.txt.gz"
# MAIN_HEADING_DATA_FILENAME = "desc2020.xml.gz"
# NUM_BASELINE_FILES = 1015
# SUBHEADING_DATA_FILENAME = "qual2020.xml.gz"
# TEST_SET_START_YEAR = 2019
# TRAIN_SET_START_YEAR = 2005
# TRAIN_END_TO_END_MODEL = False
# SENTENCEPIECE_MODEL_START_YEAR = 2016
# SMALL_TRAIN_SET_START_YEAR = 2016    # 2016 | 2005
# SUBHEADING_MODEL_DROPOUT_RATE = 0.5  # 0.5  | 0.25 (As you increase the small training set size you can reduce the dropout rate)
# USE_TEST_SET_PMIDS = False
# TEST_SET_PMIDS_FILENAME = "test_set_pmids.txt"

# User settings paper (2019 baseline)
BASELINE_FILENAME_TEMPLATE = "pubmed19n{0:04d}.xml.gz"
BASELINE_URL = "https://mbr.nlm.nih.gov/Download/Baselines/2019/"
JOURNAL_MEDLINE_FILENAME = "J_Medline_2_Jan_19.txt.gz"
MAIN_HEADING_DATA_FILENAME = "desc2019.xml.gz"
NUM_BASELINE_FILES = 972
SUBHEADING_DATA_FILENAME = "qual2019.xml.gz"
TEST_SET_START_YEAR = 2018
TRAIN_SET_START_YEAR = 2004
TRAIN_END_TO_END_MODEL = True
SENTENCEPIECE_MODEL_START_YEAR = 2015
SMALL_TRAIN_SET_START_YEAR = 2015
SUBHEADING_MODEL_DROPOUT_RATE = 0.5
USE_TEST_SET_PMIDS = True
TEST_SET_PMIDS_FILENAME = "test_set_pmids.txt"

# System settings
CRITICAL_MESH_PAIR_ID_MAPPING_FILENAME = "critical_mesh_pair_id_mapping.pkl"
CRITICAL_SUBHEADING_ID_MAPPING_FILENAME = "critical_subheading_id_mapping.pkl"
DATASET_PROPERTIES_FILENAME = "dataset_properties.pkl"
DEPLOY_DIR = "deploy"
DOWNLOADED_DATA_FILENAME_TEMPLATE = "{0:04d}.xml.gz"
ENCODING = "utf8"
END_TO_END_MODEL_DIR = "end_to_end_model"
EXTRACTED_DATA_FILENAME_TEMPLATE = "{0:04d}.json.gz"
HUMAN_INDEXING_METHOD = "Human"
JOURNAL_ID_LOOKUP_FILENAME = "journal_id_lookup.pkl"
MESH_PAIR_ID_LOOKUP_FILENAME = "mesh_pair_id_lookup.pkl"
MAIN_HEADING_ID_LOOKUP_FILENAME = "main_heading_id_lookup.pkl"
MAIN_HEADING_MODEL_DIR = "main_heading_model"
MEDLINE_DATA_DIR = "medline_data"
SENTENCEPIECE_MODEL_PREFIX = "bpe_64000_lc"
SENTENCES_FILENAME = "sentences.txt"
SUBHEADING_ID_LOOKUP_FILENAME = "subheading_id_lookup.pkl"
SUBHEADING_MODEL_DIR = "subheading_model"
TEST_BATCH_SIZE = 100
TEST_SET_FILENAME = "test_set.jsonl.gz"
TEST_SET_SIZE = 40000
TRAIN_SET_FILENAME = "train_set.jsonl.gz"
SMALL_TRAIN_SET_FILENAME = "small_train_set.jsonl.gz"
VAL_SET_FILENAME = "val_set.jsonl.gz"
VAL_SET_SIZE = 20000

CRITICAL_SUBHEADINGS = ["Q000009",
                        "Q000139",                               
                        "Q000150",
                        "Q000175",
                        "Q000000981",
                        "Q000188",
                        "Q000453",
                        "Q000209",
                        "Q000235",
                        "Q000494",
                        "Q000517",
                        "Q000532",
                        "Q000601",
                        "Q000627",
                        "Q000628",
                        "Q000633",
                        "Q000662"]