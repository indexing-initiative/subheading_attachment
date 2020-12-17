from .cnn.train import run as train_cnn
from . import config as cfg
from . import helper
import os


def run(workdir):

    encoding = cfg.ENCODING
    end_to_end_output_dir =               os.path.join(workdir, cfg.END_TO_END_MODEL_DIR)
    critical_mesh_pair_id_mapping_path =  os.path.join(workdir, cfg.CRITICAL_MESH_PAIR_ID_MAPPING_FILENAME)
    critical_subheading_id_mapping_path = os.path.join(workdir, cfg.CRITICAL_SUBHEADING_ID_MAPPING_FILENAME)
    dataset_properties_path =             os.path.join(workdir, cfg.DATASET_PROPERTIES_FILENAME)
    journal_id_lookup_path =              os.path.join(workdir, cfg.JOURNAL_ID_LOOKUP_FILENAME)
    mesh_pair_id_lookup_path =            os.path.join(workdir, cfg.MESH_PAIR_ID_LOOKUP_FILENAME)
    main_heading_id_lookup_path =         os.path.join(workdir, cfg.MAIN_HEADING_ID_LOOKUP_FILENAME)
    main_heading_output_dir =             os.path.join(workdir, cfg.MAIN_HEADING_MODEL_DIR)
    sentencepiece_model_path =            os.path.join(workdir, cfg.SENTENCEPIECE_MODEL_PREFIX + ".model")
    subheading_id_lookup_path =           os.path.join(workdir, cfg.SUBHEADING_ID_LOOKUP_FILENAME)
    subheading_output_dir =               os.path.join(workdir, cfg.SUBHEADING_MODEL_DIR)
    train_end_to_end_model =              cfg.TRAIN_END_TO_END_MODEL
    train_set_path =                      os.path.join(workdir, cfg.TRAIN_SET_FILENAME)
    small_train_set_path =                os.path.join(workdir, cfg.SMALL_TRAIN_SET_FILENAME)
    subheading_model_dropout_rate =       cfg.SUBHEADING_MODEL_DROPOUT_RATE
    val_set_path =                        os.path.join(workdir, cfg.VAL_SET_FILENAME)

    critical_mesh_pair_id_mapping =  helper.load_pickled_object(critical_mesh_pair_id_mapping_path)
    critical_subheading_id_mapping = helper.load_pickled_object(critical_subheading_id_mapping_path)
    dataset_properties =             helper.load_pickled_object(dataset_properties_path)
    journal_id_lookup =              helper.load_pickled_object(journal_id_lookup_path)
    mesh_pair_id_lookup =            helper.load_pickled_object(mesh_pair_id_lookup_path)
    main_heading_id_lookup =         helper.load_pickled_object(main_heading_id_lookup_path)
    subheading_id_lookup   =         helper.load_pickled_object(subheading_id_lookup_path)
    sp_processor =                   helper.create_sp_processor(sentencepiece_model_path) 
    train_set =                      helper.load_dataset(train_set_path, encoding)
    val_set =                        helper.load_dataset(val_set_path, encoding)

    training_resources = {
        "critical_mesh_pair_id_mapping": critical_mesh_pair_id_mapping,
        "critical_subheading_id_mapping": critical_subheading_id_mapping,
        "journal_id_lookup": journal_id_lookup,
        "mesh_pair_id_lookup": mesh_pair_id_lookup,
        "main_heading_id_lookup": main_heading_id_lookup,
        "subheading_id_lookup": subheading_id_lookup,
        "subheading_model_dropout_rate": subheading_model_dropout_rate,
        "sp_processor": sp_processor,
    }

    if train_end_to_end_model:
        print("Training end to end model...")
        train_cnn(end_to_end_output_dir, training_resources, train_set, val_set, dataset_properties, "end_to_end")

    print("Training main heading model...")
    train_cnn(main_heading_output_dir, training_resources, train_set, val_set, dataset_properties, "mainheading")
    
    del(train_set)
    small_train_set = helper.load_dataset(small_train_set_path, encoding)
 
    print("Training subheading model...")
    train_cnn(subheading_output_dir, training_resources, small_train_set, val_set, dataset_properties, "subheading")