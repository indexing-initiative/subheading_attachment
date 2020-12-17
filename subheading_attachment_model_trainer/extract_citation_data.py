from . import config as cfg
import dateutil.parser
import gzip
from . import helper
import json
import os
import re
import xml.etree.ElementTree as ET


MEDLINE_CITATION_NODE_PATH = "PubmedArticle/MedlineCitation"


def run(workdir):
   
    start_file_num = 1
    end_file_num = cfg.NUM_BASELINE_FILES

    datadir = os.path.join(workdir, cfg.MEDLINE_DATA_DIR)
    log_path = os.path.join(datadir, "extract_data_log.txt")
    downloaded_data_path_template = os.path.join(datadir, cfg.DOWNLOADED_DATA_FILENAME_TEMPLATE)
    extracted_data_path_template = os.path.join(datadir, cfg.EXTRACTED_DATA_FILENAME_TEMPLATE)
    encoding = cfg.ENCODING
    
    mesh_pair_id_lookup_path =    os.path.join(workdir, cfg.MESH_PAIR_ID_LOOKUP_FILENAME)
    main_heading_id_lookup_path = os.path.join(workdir, cfg.MAIN_HEADING_ID_LOOKUP_FILENAME)
    subheading_id_lookup_path =   os.path.join(workdir, cfg.SUBHEADING_ID_LOOKUP_FILENAME)
    mesh_pair_id_lookup    =  helper.load_pickled_object(mesh_pair_id_lookup_path)
    main_heading_id_lookup =  helper.load_pickled_object(main_heading_id_lookup_path)
    subheading_id_lookup   =  helper.load_pickled_object(subheading_id_lookup_path)
    valid_mesh_headings = { "mesh_pairs": set(mesh_pair_id_lookup.keys()),
                            "main_heading_uis": set(main_heading_id_lookup.keys()), 
                            "subheading_uis":   set(subheading_id_lookup.keys())}
   
    helper.create_dir(datadir)
    with open(log_path, "wt") as log_file: 
        for file_num in range(start_file_num, end_file_num + 1):
            print(f"{file_num}/{end_file_num}", end="\r")
            log_file.write(f"File number,{file_num},\n")
            downloaded_data_path = downloaded_data_path_template.format(file_num)
            extracted_data_path = extracted_data_path_template.format(file_num)
            with gzip.open(downloaded_data_path, "rt", encoding=encoding) as read_file, \
                 gzip.open(extracted_data_path, "wt", encoding=encoding) as write_file:
                root_node = ET.parse(read_file)
                extracted_data = _extract_data(root_node, valid_mesh_headings, log_file)
                json.dump(extracted_data, write_file, ensure_ascii=False, indent=4)
            log_file.flush()
    print(f"{end_file_num}/{end_file_num}")


def _citation_is_relevant(citation_data):
    is_relevant = (citation_data["pmid"] and 
                   citation_data["title"] and 
                   citation_data["journal_nlmid"] and 
                   citation_data["pub_year"] and 
                   citation_data["year_completed"] and
                   citation_data["mesh_headings"])
    return is_relevant


def _extract_citation_data(medline_citation_node, valid_mesh_headings, log_file):

    pmid_node = medline_citation_node.find("PMID")
    pmid = pmid_node.text.strip()
    pmid = int(pmid)

    title = ""
    title_node = medline_citation_node.find("Article/ArticleTitle") 
    title = ET.tostring(title_node, encoding="unicode", method="text")
    title = title.strip() if title is not None else ""
    
    abstract = ""
    abstract_node = medline_citation_node.find("Article/Abstract")
    if abstract_node is not None:
        abstract_text_nodes = abstract_node.findall("AbstractText")
        for abstract_text_node in abstract_text_nodes:
            if "Label" in abstract_text_node.attrib:
                if len(abstract) > 0:
                    abstract += " "
                abstract += abstract_text_node.attrib["Label"].strip() + ": "
            abstract_text = ET.tostring(abstract_text_node, encoding="unicode", method="text")
            if abstract_text is not None:
                abstract += abstract_text.strip()

    journal_nlmid_node = medline_citation_node.find("MedlineJournalInfo/NlmUniqueID")
    journal_nlmid = journal_nlmid_node.text.strip() if journal_nlmid_node is not None else None

    medlinedate_node = medline_citation_node.find("Article/Journal/JournalIssue/PubDate/MedlineDate")
    if medlinedate_node is not None:
        medlinedate_text = medlinedate_node.text.strip()
        pub_year = _extract_year_from_medlinedate(pmid, medlinedate_text, log_file)
    else:
        pub_year_node = medline_citation_node.find("Article/Journal/JournalIssue/PubDate/Year")
        pub_year = pub_year_node.text.strip()
        pub_year = int(pub_year)

    year_completed = None
    date_completed_node = medline_citation_node.find("DateCompleted")
    if date_completed_node is not None:
        year_completed = int(date_completed_node.find("Year").text.strip())
       
    mesh_headings = []
    mesh_heading_list_node = medline_citation_node.find("MeshHeadingList")
    if mesh_heading_list_node is not None:
        for mesh_heading_node in mesh_heading_list_node.findall("MeshHeading"):
            descriptor_name_node = mesh_heading_node.find("DescriptorName")
            descriptor_ui = descriptor_name_node.attrib["UI"].strip()
            if not descriptor_ui:
                continue
            if descriptor_ui not in valid_mesh_headings["main_heading_uis"]:
                log_file.write(f"Invalid descriptor ui,{descriptor_ui},{pmid}\n")
                continue
            qualifiers = []
            mesh_heading = [descriptor_ui, qualifiers]
            mesh_headings.append(mesh_heading)
            qualifier_name_nodes = mesh_heading_node.findall("QualifierName")
            if qualifier_name_nodes is not None:
                for qualifier_name_node in qualifier_name_nodes:
                    qualifier_ui = qualifier_name_node.attrib["UI"].strip()
                    if not qualifier_ui:
                        continue
                    if qualifier_ui not in valid_mesh_headings["subheading_uis"]:
                        log_file.write(f"Invalid qualifier ui,{qualifier_ui},{pmid}\n")
                        continue
                    if (descriptor_ui, qualifier_ui) not in valid_mesh_headings["mesh_pairs"]:
                        log_file.write(f"Invalid mesh pair,{descriptor_ui}:{qualifier_ui},{pmid}\n")
                        continue
                    qualifiers.append(qualifier_ui)

    medline_citation_node_attribs = medline_citation_node.attrib
    indexing_method_attrib_name = "IndexingMethod"
    indexing_method = medline_citation_node_attribs[indexing_method_attrib_name].strip() if indexing_method_attrib_name in medline_citation_node_attribs else cfg.HUMAN_INDEXING_METHOD
    
    citation_data = {
                "pmid": pmid, 
                "title": title, 
                "abstract": abstract,
                "journal_nlmid": journal_nlmid, 
                "pub_year": pub_year,
                "year_completed": year_completed,
                "mesh_headings": mesh_headings,
                "indexing_method": indexing_method,
                }

    return citation_data


def _extract_data(root_xml_node, valid_mesh_headings, log_file):
    citation_data_list = []
    for medline_citation_node in root_xml_node.findall(MEDLINE_CITATION_NODE_PATH):
         citation_data = _extract_citation_data(medline_citation_node, valid_mesh_headings, log_file)
         if _citation_is_relevant(citation_data):
            citation_data_list.append(citation_data)
    extracted_data = { "citations": citation_data_list }
    return extracted_data


def _extract_year_from_medlinedate(pmid, medlinedate_text, log_file):
    pub_year = medlinedate_text[:4]
    try:
        pub_year = int(pub_year)
    except ValueError:
        match = re.search(r"\d{4}", medlinedate_text)
        if match:
            pub_year = match.group(0)
            pub_year = int(pub_year)
            log_file.write(f"YearRe,{pub_year},{pmid}\n")
        else:
            try:
                pub_year = dateutil.parser.parse(medlinedate_text, fuzzy=True).date().year
                log_file.write(f"Fuzzy,{pub_year},{pmid}\n")
            except:
                pub_year = None
                log_file.write(f"Invalid,{medlinedate_text},{pmid}\n")
    if pub_year:
        if 1500 > pub_year > 2100:
            log_file.write(f"OutOfRange,{pub_year},{pmid}\n")
    return pub_year