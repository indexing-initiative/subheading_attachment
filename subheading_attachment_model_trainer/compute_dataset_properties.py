from . import config as cfg
import json
from . import helper
import os


def run(workdir):

    encoding = cfg.ENCODING
    critical_mesh_pair_id_mapping_path =  os.path.join(workdir, cfg.CRITICAL_MESH_PAIR_ID_MAPPING_FILENAME)
    critical_subheading_id_mapping_path = os.path.join(workdir, cfg.CRITICAL_SUBHEADING_ID_MAPPING_FILENAME)
    dataset_properties_path =             os.path.join(workdir, cfg.DATASET_PROPERTIES_FILENAME)
    journal_id_lookup_path =              os.path.join(workdir, cfg.JOURNAL_ID_LOOKUP_FILENAME)
    main_heading_id_lookup_path =         os.path.join(workdir, cfg.MAIN_HEADING_ID_LOOKUP_FILENAME)
    test_set_path =                       os.path.join(workdir, cfg.TEST_SET_FILENAME)       
    train_set_path =                      os.path.join(workdir, cfg.TRAIN_SET_FILENAME)
    val_set_path =                        os.path.join(workdir, cfg.VAL_SET_FILENAME)
    
    critical_mesh_pair_id_mapping =  helper.load_pickled_object(critical_mesh_pair_id_mapping_path)
    critical_subheading_id_mapping = helper.load_pickled_object(critical_subheading_id_mapping_path)
    journal_id_lookup =              helper.load_pickled_object(journal_id_lookup_path)
    main_heading_id_lookup =         helper.load_pickled_object(main_heading_id_lookup_path)
    test_set =                       helper.load_dataset(test_set_path, encoding)
    train_set =                      helper.load_dataset(train_set_path, encoding)
    val_set =                        helper.load_dataset(val_set_path, encoding)

    num_critical_mesh_pairs =  len(critical_mesh_pair_id_mapping)
    num_critical_subheadings = len(critical_subheading_id_mapping)
    num_journals = len(journal_id_lookup) + 1 # + 1 for unknown
    num_main_headings = len(main_heading_id_lookup)

    max_pub_year = 0
    max_year_completed = 0
    citation_count = len(train_set) + len(val_set) + len(test_set)
    for idx, citation in enumerate(train_set + val_set + test_set):
        print(f"{idx}/{citation_count}", end="\r")
        citation = json.loads(citation)
        pub_year = citation["pub_year"]
        year_completed = citation["year_completed"]
        if pub_year > max_pub_year:
            max_pub_year = pub_year
        if year_completed > max_year_completed:
            max_year_completed = year_completed
    print(f"{citation_count}/{citation_count}")

    dataset_properties = {
                        "max_pub_year": max_pub_year, 
                        "max_year_completed": max_year_completed, 
                        "num_critical_mesh_pairs": num_critical_mesh_pairs,
                        "num_critical_subheadings": num_critical_subheadings,
                        "num_journals": num_journals,
                        "num_main_headings": num_main_headings, 
                        }
    print(dataset_properties)
    helper.pickle_dump(dataset_properties, dataset_properties_path)
    return dataset_properties