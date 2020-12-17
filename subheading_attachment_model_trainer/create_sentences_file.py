from . import config as cfg
from . import helper
import json
import os
import spacy
from spacy.lang.en import English


def run(workdir):

    start_year = cfg.SENTENCEPIECE_MODEL_START_YEAR
    train_dataset_path = os.path.join(workdir, cfg.TRAIN_SET_FILENAME)
    encoding = cfg.ENCODING
    sentences_path = os.path.join(workdir, cfg.SENTENCES_FILENAME)

    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)

    train_dataset = helper.load_dataset(train_dataset_path, encoding)
    train_set_size = len(train_dataset)

    with open(sentences_path, "wt", encoding=encoding) as write_file:
        
        for idx, citation in enumerate(train_dataset):
            citation = json.loads(citation)
            if citation["pub_year"] < start_year:
                continue
            print(f"{idx}/{train_set_size}", end="\r")
            
            title, abstract = citation["title"], citation["abstract"]
            title = title.strip()
            if title[0] == "[" and title[-2:] == "].":
                title = title[1:-2] + "."
            title = title.strip()
            abstract = abstract.strip()
            if title:
                try:
                    citation_text = ""
                    doc = nlp(title)
                    for sent in doc.sents:
                        citation_text += sent.text + "\n"
                    if abstract:
                        doc = nlp(abstract)
                        for sent in doc.sents:
                            citation_text += sent.text + "\n"
                    write_file.write(citation_text)
                except Exception:
                    print("Exception")
    
    print(f"{train_set_size}/{train_set_size}")