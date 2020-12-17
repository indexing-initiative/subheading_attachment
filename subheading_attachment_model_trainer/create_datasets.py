from . import config as cfg
import json
import gzip
import os
import random


def run(workdir):
    encoding = cfg.ENCODING
    extracted_data_path_template = os.path.join(workdir, cfg.MEDLINE_DATA_DIR, cfg.EXTRACTED_DATA_FILENAME_TEMPLATE)
    num_data_files = cfg.NUM_BASELINE_FILES
    train_set_start_year = cfg.TRAIN_SET_START_YEAR
    small_train_set_start_year = cfg.SMALL_TRAIN_SET_START_YEAR
    test_set_start_year = cfg.TEST_SET_START_YEAR
    test_set_size = cfg.TEST_SET_SIZE
    use_test_set_pmids = cfg.USE_TEST_SET_PMIDS
    val_set_size = cfg.VAL_SET_SIZE

    train_set_path = os.path.join(workdir, cfg.TRAIN_SET_FILENAME)
    small_train_set_path = os.path.join(workdir, cfg.SMALL_TRAIN_SET_FILENAME)
    val_set_path =   os.path.join(workdir, cfg.VAL_SET_FILENAME)
    test_set_path =  os.path.join(workdir, cfg.TEST_SET_FILENAME)
    test_set_pmids_path = os.path.join(workdir, cfg.TEST_SET_PMIDS_FILENAME)

    dataset, pub_year_lookup = _create_dataset(extracted_data_path_template, encoding, num_data_files, train_set_start_year)

    provided_test_set_pmids = None
    if use_test_set_pmids:
        provided_test_set_pmids  = { int(line.strip()) for line in open(test_set_pmids_path) if len(line.strip()) > 0 }
    train_set_pmids, small_train_set_pmids, val_set_pmids, test_set_pmids = _split_dataset(pub_year_lookup, use_test_set_pmids, provided_test_set_pmids, 
                                                                                           test_set_start_year, small_train_set_start_year, 
                                                                                           test_set_size, val_set_size)

    print(f"Train set size: {len(train_set_pmids)}")
    print(f"Small train set size: {len(small_train_set_pmids)}")
    print(f"Validation set size: {len(val_set_pmids)}")
    print(f"Test set size: {len(test_set_pmids)}")

    print("Saving train set...")
    _save_dataset(dataset, train_set_pmids, train_set_path, encoding)
    print("Saving small train set...")
    _save_dataset(dataset, small_train_set_pmids, small_train_set_path, encoding)
    print("Saving validation set...")
    _save_dataset(dataset, val_set_pmids, val_set_path, encoding)
    print("Saving test set...")
    _save_dataset(dataset, test_set_pmids, test_set_path, encoding)


def _create_dataset(extracted_data_path_template, encoding, num_files, min_pub_year):
    dataset = {}
    pub_year_lookup = {}
    for file_num in range(1, num_files + 1):
        print(f"{file_num}/{num_files}", end="\r")
        extracted_data_path = extracted_data_path_template.format(file_num)
        with gzip.open(extracted_data_path, "rt", encoding=encoding) as read_file: 
            data = json.load(read_file)
            for citation in data["citations"]:
                indexing_method = citation["indexing_method"]
                pub_year = citation["pub_year"]
                if indexing_method == cfg.HUMAN_INDEXING_METHOD and pub_year >= min_pub_year:
                    pmid = citation["pmid"]
                    del citation["indexing_method"]
                    citation_json = json.dumps(citation, ensure_ascii=False, indent=None)
                    dataset[pmid] = citation_json
                    pub_year_lookup[pmid] = pub_year
    print(f"{num_files}/{num_files}")
    return dataset, pub_year_lookup


def _save_dataset(dataset, pmids, path, encoding):
    with gzip.open(path, "wt", encoding=encoding) as write_file:
        for pmid in pmids:
            if pmid in dataset:
                pmid_json = dataset[pmid]
                write_file.write(pmid_json)
                write_file.write("\n")


def _split_dataset(pub_year_lookup, use_test_set_pmids, provided_test_set_pmids, test_set_start_year, small_train_set_start_year, test_set_size, val_set_size):
    
    if use_test_set_pmids:
        val_set_candidates = []
        train_set_pmids = []
        for pmid, pub_year in pub_year_lookup.items():
            if pmid not in provided_test_set_pmids:
                if pub_year >= test_set_start_year:
                    val_set_candidates.append(pmid)
                else:
                    train_set_pmids.append(pmid)
        random.shuffle(val_set_candidates)

        test_set_pmids = list(provided_test_set_pmids)
        val_set_pmids = val_set_candidates[:val_set_size]
        remaining_pmids = val_set_candidates[val_set_size:]
        train_set_pmids.extend(remaining_pmids)
    else:
        test_set_candidates = []
        train_set_pmids = []
        for pmid, pub_year in pub_year_lookup.items():
            if pub_year >= test_set_start_year:
                test_set_candidates.append(pmid)
            else:
                train_set_pmids.append(pmid)
        random.shuffle(test_set_candidates)

        test_val_set_size = test_set_size + val_set_size
        test_set_pmids = test_set_candidates[:test_set_size]
        val_set_pmids = test_set_candidates[test_set_size:test_val_set_size]
        remaining_pmids = test_set_candidates[test_val_set_size:]
        train_set_pmids.extend(remaining_pmids)
    
    small_train_set_pmids = [ pmid for pmid in train_set_pmids if pub_year_lookup[pmid] >= small_train_set_start_year]

    random.shuffle(train_set_pmids)
    random.shuffle(small_train_set_pmids)
    random.shuffle(val_set_pmids)
    random.shuffle(test_set_pmids)
    return train_set_pmids, small_train_set_pmids, val_set_pmids, test_set_pmids