from . import compute_dataset_properties
from . import create_critical_mesh_pair_id_mapping
from . import create_critical_subheading_id_mapping
from . import create_datasets
from . import create_journal_id_lookup
from . import create_mesh_pair_id_lookup
from . import create_main_heading_id_lookup
from . import create_sentences_file
from . import create_subheading_id_lookup
from . import download_medline_baseline
from . import extract_citation_data
from . import train_models
from . import train_sentencepiece_model
from . import wrap_models


def run(workdir):
    print("Creating main heading id lookup...")
    create_main_heading_id_lookup.run(workdir)
    print("Creating subheading id lookup...")
    create_subheading_id_lookup.run(workdir)
    print("Creating mesh pair id lookup...")
    create_mesh_pair_id_lookup.run(workdir)
    print("Creating critical subheading id mapping...")
    create_critical_subheading_id_mapping.run(workdir)
    print("Creating critical mesh pair id mapping...")
    create_critical_mesh_pair_id_mapping.run(workdir)
    print("Create journal id lookup...")
    create_journal_id_lookup.run(workdir)

    print("Downloading MEDLINE baseline...")
    download_medline_baseline.run(workdir)
    print("Extracting citation data...")
    extract_citation_data.run(workdir)
    print("Creating datasets...")
    create_datasets.run(workdir)
    print("Computing dataset properties...")
    compute_dataset_properties.run(workdir)

    print("Creating sentences file...")
    create_sentences_file.run(workdir)
    print("Training SentencePiece model...")
    train_sentencepiece_model.run(workdir)


    print("Training models...")
    train_models.run(workdir)
    print("Wrapping models...")
    wrap_models.run(workdir)