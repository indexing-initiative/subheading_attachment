from .  import data_helper
from .model import Model
import os
from . import settings


def run(output_dir, training_resources, train_set, val_set, dataset_properties, model_type):
    config = _get_config(training_resources, dataset_properties, model_type)
    train_gen, dev_gen = _get_generators(config, training_resources, train_set, val_set, model_type)
    model = Model()
    model.build(config.model)
    model.fit(config, train_gen, dev_gen, output_dir)


def _get_config(training_resources, dataset_properties, model_type):
    config = settings.get_config()
    model_cfg = config.model
    pp_config = config.preprocessing

    dropout_rate = None
    has_main_heading_input = False
    num_labels = None
    num_main_headings = dataset_properties["num_main_headings"]

    if model_type == "end_to_end":
        dropout_rate = 0.25
        num_labels = dataset_properties["num_critical_mesh_pairs"]
    elif model_type == "mainheading":
        dropout_rate = 0.05
        num_labels = num_main_headings
    elif model_type == "subheading":
        dropout_rate = training_resources["subheading_model_dropout_rate"]
        has_main_heading_input = True
        num_labels = dataset_properties["num_critical_subheadings"]
    else:
        raise ValueError(f"model_type, {model_type}, not recognised.")

    model_cfg.dropout_rate = dropout_rate
    model_cfg.has_main_heading_input = has_main_heading_input
    model_cfg.num_journals = dataset_properties["num_journals"]

    pp_config.max_pub_year = dataset_properties["max_pub_year"]
    pp_config.max_year_completed = dataset_properties["max_year_completed"]
    pp_config.num_labels = num_labels
    pp_config.num_main_headings = num_main_headings

    return config


def _get_generators(config, training_resources, train_set, val_set, model_type):

    pp_config = config.preprocessing
    train_batch_size = config.train.batch_size
    dev_batch_size = config.train.dev_batch_size
    train_limit = config.train.train_limit
    dev_limit = config.train.dev_limit

    if model_type == "end_to_end":
        train_gen = data_helper.EndToEndGenerator(pp_config, training_resources, train_set, train_batch_size, train_limit)
        dev_gen =   data_helper.EndToEndGenerator(pp_config, training_resources, val_set, dev_batch_size, dev_limit)
    elif model_type == "mainheading":
        train_gen = data_helper.MainHeadingGenerator(pp_config, training_resources, train_set, train_batch_size, train_limit)
        dev_gen =   data_helper.MainHeadingGenerator(pp_config, training_resources, val_set, dev_batch_size, dev_limit)
    elif model_type == "subheading":
        train_gen = data_helper.SubheadingGenerator(pp_config, training_resources, train_set, train_batch_size, train_limit)
        dev_gen =   data_helper.SubheadingGenerator(pp_config, training_resources, val_set, dev_batch_size, dev_limit)   
    else:
        raise ValueError(f"model_type, {model_type}, not recognised.")

    return train_gen, dev_gen