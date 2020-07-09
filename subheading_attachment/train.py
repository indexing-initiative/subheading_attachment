from .  import data_helper
from .model import Model
import os
from . import settings
import time

config = settings.get_config()
db_config = config.database
pp_config = config.preprocessing
ofs_config = config.train.optimize_fscore_threshold
model_config = config.model

subdir = str(int(time.time()))
output_dir = os.path.join(config.root_dir, subdir)
print(output_dir)
    
train_set_ids, dev_set_ids = data_helper.load_cross_validation_ids(config.cross_val)
sp_processor = data_helper.create_sp_processor(pp_config.sentencepiece_model_path)
label_id_mapping = pp_config.label_id_mapping
train_gen = data_helper.DatabaseGenerator(db_config, 
                                          pp_config, 
                                          sp_processor, 
                                          label_id_mapping, 
                                          train_set_ids, 
                                          config.train.batch_size, 
                                          config.train.train_limit, 
                                          config.train.max_avg_desc_per_citation)
dev_gen =   data_helper.DatabaseGenerator(db_config, 
                                          pp_config, 
                                          sp_processor, 
                                          label_id_mapping, 
                                          dev_set_ids, 
                                          config.train.dev_batch_size, 
                                          config.train.dev_limit)

model = Model()
model.build(model_config)
model.fit(config, train_gen, dev_gen, output_dir)