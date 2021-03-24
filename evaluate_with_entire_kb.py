from typing import Dict, Iterable, List, Tuple
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
import torch
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from utils import build_one_flag_loader, build_vocab
from dataset_reader import EntitiesInKBLoader
from parameteres import Biencoder_params
from utils import build_vocab, build_data_loaders, build_one_flag_loader, emb_returner, build_trainer
from encoder import Pooler_for_mention, Pooler_for_cano_and_def

def evaluate_with_kb(params, entity_encoder):
    ds = EntitiesInKBLoader(params)
    entities = ds._read()
    vocab = build_vocab(entities)

    entity_loader = build_one_flag_loader(params, entities)
    entity_loader.index_with(vocab)

if __name__ == '__main__':
    config = Biencoder_params()
    params = config.opts
    _, __, embedder = emb_returner(config=params)
    mention_encoder, entity_encoder = Pooler_for_mention(params, embedder), Pooler_for_cano_and_def(params, embedder)

    evaluate_with_kb(params, entity_encoder)