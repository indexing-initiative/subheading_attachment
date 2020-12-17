from . import config as cfg
from . import helper
import os
import time
import urllib.request


MAX_RETRYS = 5


def run(workdir):
    start_file_num = 1
    end_file_num = cfg.NUM_BASELINE_FILES
    
    datadir = os.path.join(workdir, cfg.MEDLINE_DATA_DIR)
    url_template = os.path.join(cfg.BASELINE_URL, cfg.BASELINE_FILENAME_TEMPLATE)
    save_path_template = os.path.join(datadir, cfg.DOWNLOADED_DATA_FILENAME_TEMPLATE)

    helper.create_dir(datadir)
    for file_num in range(start_file_num, end_file_num + 1):
        url = url_template.format(file_num)
        save_path = save_path_template.format(file_num)
        print(f"{file_num}/{end_file_num}", end="\r")
        num_retrys = 0
        while True:
            try:
                urllib.request.urlretrieve(url, save_path)
                break
            except:
                if num_retrys >= MAX_RETRYS:
                    raise Exception(f"Aborting: failed to download file number {file_num}")
                num_retrys += 1
                print(f"Retrying download: {num_retrys}/{MAX_RETRYS}")
                time.sleep(60.)
                continue

    print(f"{file_num}/{end_file_num}")

    return end_file_num