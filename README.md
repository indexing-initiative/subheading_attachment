# Subheading Attachment

This repository contains the code for the paper "Automatic MeSH Indexing: Revisiting the Subheading Attachment Problem" by A. R. Rae et al.

Usage:

```
conda create -n subheading_attachment --file requirements.txt
conda activate subheading_attachment
python -m subheading_attachment.train --model_type=end_to_end
python -m subheading_attachment.train --model_type=mainheading
python -m subheading_attachment.train --model_type=subheading
```