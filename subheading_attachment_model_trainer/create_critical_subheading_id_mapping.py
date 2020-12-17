from . import config as cfg
from . import helper
import os


def run(workdir):
    subheading_id_lookup_path = os.path.join(workdir, cfg.SUBHEADING_ID_LOOKUP_FILENAME)
    critical_subheading_id_mapping_path = os.path.join(workdir, cfg.CRITICAL_SUBHEADING_ID_MAPPING_FILENAME)

    subheading_id_lookup = helper.load_pickled_object(subheading_id_lookup_path)
   
    mapping = {}
    for idx, qui in enumerate(cfg.CRITICAL_SUBHEADINGS):
        org_id = subheading_id_lookup[qui]
        mapped_id = idx + 1
        mapping[org_id] = mapped_id

    helper.pickle_dump(mapping, critical_subheading_id_mapping_path)