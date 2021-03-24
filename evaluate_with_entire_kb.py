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
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
from tqdm import tqdm
import pdb
from kb_loader import KBIndexerWithFaiss
from model import BiencoderNNSearchEvaluator
from allennlp.training.util import evaluate

class KBEntityEmbEncoder(Predictor):
    def predict(self, entitiy_unique_id: int) -> JsonDict:
        return self.predict_json({"entity_unique_id": entitiy_unique_id})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        entity_unique_id = json_dict["entity_unique_id"]
        return self._dataset_reader.text_to_instance(entity_unique_id=entity_unique_id)


def evaluate_with_kb(params, mention_encoder, model, dev_loader, test_loader):
    ds = EntitiesInKBLoader(params)
    entities = ds._read()
    entity_ids = ds.get_entity_ids()
    vocab = build_vocab(entities)

    entity_loader = build_one_flag_loader(params, entities)
    entity_loader.index_with(vocab)
    predictor = KBEntityEmbEncoder(model, ds)

    entity_idx2emb = {}

    print('===Encoding All Entities from Fine-Tuned Entity Encoder===')
    for entity_id in tqdm(entity_ids):
        its_emb = predictor.predict(entity_id)['encoded_entities']
        entity_idx2emb.update({entity_id: its_emb})
        if params.debug and len(entity_idx2emb) == 300:
            break

    kb = KBIndexerWithFaiss(
        config=params, entity_idx2emb=entity_idx2emb
    )

    evaluate_model = BiencoderNNSearchEvaluator(params, mention_encoder, vocab, kb)

    dev_eval_result = evaluate(model=evaluate_model, data_loader=dev_loader, cuda_device=0,
                               batch_weight_key="")

if __name__ == '__main__':
    config = Biencoder_params()
    params = config.opts
    _, __, embedder = emb_returner(config=params)
    mention_encoder, entity_encoder = Pooler_for_mention(params, embedder), Pooler_for_cano_and_def(params, embedder)

    evaluate_with_kb(params, entity_encoder)