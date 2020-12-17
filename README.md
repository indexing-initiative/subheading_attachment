# Subheading Attachment Model Trainer

This package trains the subheading attachment models described in "Automatic MeSH Indexing: Revisting the Subheading Attachment Problem" by A. R. Rae et al.

## System Requirements

- 64GB RAM (Required)
- 64GB disk space (Required)
- NVIDIA 16GB GPU (Required)
- Multi-core CPU (Recommended) (Tested with 14 CPU cores)

## Install

1) Create an Anconda environment:

```
conda create -n trainer_env --file requirements.txt
conda activate trainer_env
pip install tensorflow-text==2.2.0
pip install tensorflow-gpu==2.2.0
pip install tensorflow-serving-api==2.2
```
2) To install from source, run the following command in the package root directory:
```
    pip install .
```

## Train

1) Create a working directory.
2) Copy all files from from ./input_data to the working directory.
3) From the working direcory run:
```
python -m subheading_attachment_model_trainer.train
```
Trained models are saved to the working directory in "end_to_end_model", "main_heading_model", and "subheading_model" folders.

## Test Chained Method

1) Start tensorflow serving using Docker:
```
sudo docker run -d -p 8500:8500 -v /path/to/workdir/deploy:/serving/sh-pred/1 tensorflow/serving:2.2.0 --model_name=sh-pred --model_base_path=/serving/sh-pred
```
2) From the working directory run:
```
python -m subheading_attachment_model_trainer.test
```
The script prints the F1 score performance on the test set. It is expected to be about 0.44.