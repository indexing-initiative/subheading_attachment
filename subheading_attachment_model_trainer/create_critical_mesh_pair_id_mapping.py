from . import config as cfg
from . import helper
import os


def run(workdir):
    mesh_pair_id_lookup_path = os.path.join(workdir, cfg.MESH_PAIR_ID_LOOKUP_FILENAME)
    critical_mesh_pair_id_mapping_path = os.path.join(workdir, cfg.CRITICAL_MESH_PAIR_ID_MAPPING_FILENAME)
    critical_subheadings = set(cfg.CRITICAL_SUBHEADINGS)

    mesh_pair_id_lookup = helper.load_pickled_object(mesh_pair_id_lookup_path)

    mapping = {}
    _id = 1
    for key, org_id in mesh_pair_id_lookup.items():
        dui, qui = key
        if qui in critical_subheadings:
            mapping[org_id] = _id
            _id += 1
            
    helper.pickle_dump(mapping, critical_mesh_pair_id_mapping_path)