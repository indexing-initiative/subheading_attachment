from . import config as cfg
import gzip
from . import helper
import os
import xml.etree.ElementTree as ET


def run(workdir):
    encoding = cfg.ENCODING
    mesh_pair_id_lookup_path =    os.path.join(workdir, cfg.MESH_PAIR_ID_LOOKUP_FILENAME)
    main_heading_data_path =      os.path.join(workdir, cfg.MAIN_HEADING_DATA_FILENAME)
    null_subheading = str(None)

    _id = 1
    mesh_pair_id_lookup = {}
    with gzip.open(main_heading_data_path, "rt", encoding=encoding) as read_file:
        root_node = ET.parse(read_file)
        for desc_record_node in root_node.findall("DescriptorRecord"):
            desc_ui = desc_record_node.find("DescriptorUI").text.strip()
            mesh_pair_id_lookup[(desc_ui, null_subheading)] = _id
            _id += 1
            allowable_qualifier_list_node = desc_record_node.find("AllowableQualifiersList")
            if allowable_qualifier_list_node is not None:
                for allowable_qualifier_node in allowable_qualifier_list_node.findall("AllowableQualifier"):
                    qualifier_ui = allowable_qualifier_node.find("QualifierReferredTo/QualifierUI").text.strip()
                    mesh_pair_id_lookup[(desc_ui, qualifier_ui)] = _id
                    _id += 1

    helper.pickle_dump(mesh_pair_id_lookup, mesh_pair_id_lookup_path)