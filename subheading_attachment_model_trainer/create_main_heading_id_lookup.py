from . import config as cfg
import gzip
from . import helper
import os
import xml.etree.ElementTree as ET


def run(workdir):
    encoding = cfg.ENCODING
    main_heading_data_path =      os.path.join(workdir, cfg.MAIN_HEADING_DATA_FILENAME)
    main_heading_id_lookup_path = os.path.join(workdir, cfg.MAIN_HEADING_ID_LOOKUP_FILENAME)

    main_heading_id_lookup = _create_main_heading_id_lookup(main_heading_data_path, encoding)
    helper.pickle_dump(main_heading_id_lookup, main_heading_id_lookup_path)


def _create_main_heading_id_lookup(main_heading_data_path, encoding):
    lookup = {}
    _id = 1
    with gzip.open(main_heading_data_path, "rt", encoding=encoding) as read_file:
        root_node = ET.parse(read_file)
        for record_node in root_node.findall("DescriptorRecord"):
            ui = record_node.find("DescriptorUI").text.strip()
            lookup[ui] = _id
            _id += 1
    return lookup