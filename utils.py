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
import copy
from allennlp.training.util import evaluate


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)


def build_data_loaders(config,
    train_data: List[Instance],
    dev_data: List[Instance],
    test_data: List[Instance]) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_loader = SimpleDataLoader(train_data, config.batch_size_for_train, shuffle=True)
    dev_loader = SimpleDataLoader(dev_data, config.batch_size_for_eval, shuffle=False)
    test_loader = SimpleDataLoader(test_data, config.batch_size_for_eval, shuffle=False)

    return train_loader, dev_loader, test_loader

def build_one_flag_loader(config,
                          data: List[Instance]) -> DataLoader:
    loader = SimpleDataLoader(data, config.batch_size_for_eval, shuffle=False)

    return loader

def build_trainer(
    config,
    model: Model,
    train_loader: DataLoader,
    dev_loader: DataLoader,
) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters, lr=config.lr)
    if torch.cuda.is_available():
        model.cuda()
    trainer = GradientDescentTrainer(
        model=model,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=config.num_epochs,
        patience=config.patience,
        optimizer=optimizer,
        serialization_dir=config.serialization_dir,
        cuda_device=0 if torch.cuda.is_available() else -1
    )

    return trainer

def emb_returner(config):
    if config.bert_name == 'bert-base-uncased':
        huggingface_model = 'bert-base-uncased'
    elif config.bert_name == 'biobert':
        huggingface_model = './biobert/'
    else:
        huggingface_model = 'dummy'
        print(config.bert_name,'are not supported')
        exit()
    bert_embedder = PretrainedTransformerEmbedder(model_name=huggingface_model)
    return bert_embedder, bert_embedder.get_output_dim(), BasicTextFieldEmbedder({'tokens': bert_embedder})


def candidate_recall_evaluator(dev_or_test: str, model, params, data_loader):
    model.mention_idx2candidate_entity_idxs = copy.copy({})
    evaluate(model=model, data_loader=data_loader, cuda_device=0, batch_weight_key="")
    r1, r5, r10, r50 = 0, 0, 0, 0
    for _, its_candidate_and_gold in model.mention_idx2candidate_entity_idxs.items():
        candidate_entity_idxs = its_candidate_and_gold['candidate_entity_idx']
        gold_idx = its_candidate_and_gold['gold_entity_idx']

        if gold_idx in candidate_entity_idxs and candidate_entity_idxs.index(gold_idx) == 0:
            r1 += 1
            r5 += 1
            r10 += 1
            r50 += 1
            continue

        elif gold_idx in candidate_entity_idxs and candidate_entity_idxs.index(gold_idx) < 5:
            r5 += 1
            r10 += 1
            r50 += 1
            continue

        elif gold_idx in candidate_entity_idxs and candidate_entity_idxs.index(gold_idx) < 10:
            r10 += 1
            r50 += 1
            continue

        elif gold_idx in candidate_entity_idxs and candidate_entity_idxs.index(gold_idx) < 50:
            r50 += 1
            continue

        else:
            continue

    r1 = r1 / len(model.mention_idx2candidate_entity_idxs)
    r5 = r5 / len(model.mention_idx2candidate_entity_idxs)
    r10 = r10 / len(model.mention_idx2candidate_entity_idxs)
    r50 = r50 / len(model.mention_idx2candidate_entity_idxs)

    print('{}'.format(dev_or_test), 'evaluation result')
    print('recall@{}'.format(params.how_many_top_hits_preserved), round(r50 * 100, 3), '%')
    print('detail recall@1, @5, @10, @50',
          round(r1 * 100, 3), '%', round(r5 * 100, 3), '%', round(r10 * 100, 3), '%', round(r50 * 100, 3), '%',
          )