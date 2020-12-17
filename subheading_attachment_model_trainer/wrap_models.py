from .cnn.wrap_models import run as wrap
from . import config as cfg
from . import helper
import os


def run(workdir):

    critical_subheading_id_mapping_path = os.path.join(workdir, cfg.CRITICAL_SUBHEADING_ID_MAPPING_FILENAME)
    dataset_properties_path =             os.path.join(workdir, cfg.DATASET_PROPERTIES_FILENAME)
    deploy_dir =                          os.path.join(workdir, cfg.DEPLOY_DIR)
    journal_id_lookup_path =              os.path.join(workdir, cfg.JOURNAL_ID_LOOKUP_FILENAME)
    main_heading_id_lookup_path =         os.path.join(workdir, cfg.MAIN_HEADING_ID_LOOKUP_FILENAME)
    main_heading_output_dir =             os.path.join(workdir, cfg.MAIN_HEADING_MODEL_DIR)
    sentencepiece_model_path =            os.path.join(workdir, cfg.SENTENCEPIECE_MODEL_PREFIX + ".model")
    subheading_id_lookup_path =           os.path.join(workdir, cfg.SUBHEADING_ID_LOOKUP_FILENAME)
    subheading_output_dir =               os.path.join(workdir, cfg.SUBHEADING_MODEL_DIR)

    critical_subheading_id_mapping = helper.load_pickled_object(critical_subheading_id_mapping_path)
    dataset_properties =             helper.load_pickled_object(dataset_properties_path)
    journal_id_lookup =              helper.load_pickled_object(journal_id_lookup_path)
    main_heading_id_lookup =         helper.load_pickled_object(main_heading_id_lookup_path)
    sp_model_proto =                 helper.load_sentencepiece_model_proto(sentencepiece_model_path)
    subheading_id_lookup =           helper.load_pickled_object(subheading_id_lookup_path)
  
    helper.create_dir(deploy_dir)

    wrap_model_resources = {
        "critical_subheading_id_mapping": critical_subheading_id_mapping,
        "journal_id_lookup": journal_id_lookup,
        "main_heading_id_lookup": main_heading_id_lookup,
        "sp_model_proto": sp_model_proto,
        "subheading_id_lookup": subheading_id_lookup,
    }

    wrap(deploy_dir, main_heading_output_dir, subheading_output_dir, wrap_model_resources, dataset_properties)