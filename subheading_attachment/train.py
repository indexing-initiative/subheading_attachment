import argparse
from .  import data_helper
from .model import Model
import os
from . import settings
import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='end_to_end')
    args = parser.parse_args()
    return args


def get_config(model_type):
    config = settings.get_config()

    dropout_rate = None
    has_desc_input = False
    label_id_mapping_filename = None
    num_labels = None
    train_set_filename = 'train_set_db_ids.txt'
   
    if model_type == 'end_to_end':
        dropout_rate = 0.25
        label_id_mapping_filename  = 'mesh_topic_id_mapping.pkl'
        num_labels = 122542
    elif model_type == 'mainheading':
        dropout_rate = 0.05
        num_labels = 29351
    elif model_type == 'subheading':
        dropout_rate = 0.5
        has_desc_input = True
        label_id_mapping_filename  = 'mesh_qualifier_id_mapping.pkl'
        num_labels = 17
        train_set_filename = 'train_recent_subset_db_ids.txt'
    else:
        raise ValueError(f'model_type, {model_type}, not recognised.')

    config.model.dropout_rate = dropout_rate
    config.preprocessing.label_id_mapping_path = os.path.join(config.data_dir, label_id_mapping_filename) if label_id_mapping_filename else None
    config.preprocessing.num_labels = num_labels
    config.model.has_desc_input = has_desc_input
    config.cross_val.train_set_ids_path = os.path.join(config.data_dir, train_set_filename)

    return config


def get_generators(config, model_type):

    db_config = config.database
    pp_config = config.preprocessing
    sp_processor = data_helper.create_sp_processor(config.preprocessing.sentencepiece_model_path)
    label_id_mapping = data_helper.load_pickled_object(pp_config.label_id_mapping_path) if pp_config.label_id_mapping_path else None
    train_set_ids, dev_set_ids = data_helper.load_cross_validation_ids(config.cross_val)
    train_batch_size = config.train.batch_size
    train_limit = config.train.train_limit
    max_avg_desc_per_citation = config.train.max_avg_desc_per_citation
    dev_batch_size = config.train.dev_batch_size
    dev_limit = config.train.dev_limit

    if model_type == 'end_to_end':
        train_gen = data_helper.MeshPairDatabaseGenerator(db_config, pp_config, sp_processor, label_id_mapping, train_set_ids, train_batch_size, train_limit)
        dev_gen =   data_helper.MeshPairDatabaseGenerator(db_config, pp_config, sp_processor, label_id_mapping, dev_set_ids, dev_batch_size, dev_limit)
    elif model_type == 'mainheading':
        train_gen = data_helper.MainheadingDatabaseGenerator(db_config, pp_config, sp_processor, label_id_mapping, train_set_ids, train_batch_size, train_limit)
        dev_gen =   data_helper.MainheadingDatabaseGenerator(db_config, pp_config, sp_processor, label_id_mapping, dev_set_ids, dev_batch_size, dev_limit)
    elif model_type == 'subheading':
        train_gen = data_helper.SubheadingDatabaseGenerator(db_config, pp_config, sp_processor, label_id_mapping, train_set_ids, train_batch_size, train_limit, max_avg_desc_per_citation)
        dev_gen =   data_helper.SubheadingDatabaseGenerator(db_config, pp_config, sp_processor, label_id_mapping, dev_set_ids, dev_batch_size, dev_limit)
    else:
        raise ValueError(f'model_type, {model_type}, not recognised.')

    return train_gen, dev_gen
    

def main(args):

    model_type = args.model_type

    config = get_config(model_type)

    subdir = str(int(time.time()))
    output_dir = os.path.join(config.root_dir, subdir)
    print(output_dir)
        
    train_gen, dev_gen = get_generators(config, model_type)

    model = Model()
    model.build(config.model)
    model.fit(config, train_gen, dev_gen, output_dir)


if __name__ == '__main__':
    args = get_args()
    main(args)