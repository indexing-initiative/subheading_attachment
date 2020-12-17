from . import config as cfg
import gzip
from . import helper
import os


LINES_PER_JOURNAL = 8


def run(workdir):
    encoding = cfg.ENCODING
    journal_medline_path =   os.path.join(workdir, cfg.JOURNAL_MEDLINE_FILENAME)
    journal_id_lookup_path = os.path.join(workdir, cfg.JOURNAL_ID_LOOKUP_FILENAME)
   
    journal_id_lookup = _create_lookup(journal_medline_path, encoding)
    helper.pickle_dump(journal_id_lookup, journal_id_lookup_path)


def _create_lookup(journal_medline_path, encoding):
    with gzip.open(journal_medline_path, "rt", encoding=encoding) as read_file:
        lines = read_file.readlines()

    line_count = len(lines)
    journal_count = line_count // LINES_PER_JOURNAL
    lookup = {}
    for idx in range(journal_count):
        start_line = LINES_PER_JOURNAL*idx
        nlmid =      lines[start_line + 7].strip()[7:].strip()
        journal_id = idx + 1
        lookup[nlmid] = journal_id
    return lookup