# Subheading Attachment

This repository contains the code for the paper "Automatic MeSH Indexing: Revisiting the Subheading Attachment Problem" by A. R. Rae et al.

# Setup

1. The code reads MEDLINE data from a database. To setup a database follow the instructions in ./database/README.md

2. Set the root directory and database host in ./subheading_attachment/machine_settings.py

3. Set the database user and password in ./subheading_attachment/settings.py

4. Copy the ./input_data folder to your root directory.

5. Create a folder named "runs" in your root directory.


# Usage

```
conda create -n subheading_attachment --file requirements.txt
conda activate subheading_attachment
pip install sentencepiece
python -m subheading_attachment.train --model_type=end_to_end
python -m subheading_attachment.train --model_type=mainheading
python -m subheading_attachment.train --model_type=subheading
```