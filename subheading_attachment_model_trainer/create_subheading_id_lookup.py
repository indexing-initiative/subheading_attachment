from . import config as cfg
import gzip
from . import helper
import os
import xml.etree.ElementTree as ET


def run(workdir):
    encoding = cfg.ENCODING
    subheading_data_path =      os.path.join(workdir, cfg.SUBHEADING_DATA_FILENAME)
    subheading_id_lookup_path = os.path.join(workdir, cfg.SUBHEADING_ID_LOOKUP_FILENAME)

    subheading_id_lookup = _create_subheading_id_lookup(subheading_data_path, encoding)
    helper.pickle_dump(subheading_id_lookup, subheading_id_lookup_path)


def _create_subheading_id_lookup(subheading_data_path, encoding):
    lookup = {}
    lookup[str(None)] = 1
    _id = 2
    with gzip.open(subheading_data_path, "rt", encoding=encoding) as read_file:
        root_node = ET.parse(read_file)
        for record_node in root_node.findall("QualifierRecord"):
            ui = record_node.find("QualifierUI").text.strip()
            lookup[ui] = _id
            _id += 1 
    return lookup
    