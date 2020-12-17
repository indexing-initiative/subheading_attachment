from . import config as cfg
import os
import sentencepiece as spm

def run(workdir):
    sentences_path = os.path.join(workdir, cfg.SENTENCES_FILENAME)
    spm.SentencePieceTrainer.Train(f"--input={sentences_path} --model_prefix={cfg.SENTENCEPIECE_MODEL_PREFIX} --vocab_size=64000 --character_coverage=1.0 --model_type=bpe --shuffle_input_sentence=True --unk_id=1 --pad_id=0 --bos_id=2 --eos_id=3 --normalization_rule_name=nmt_nfkc_cf")