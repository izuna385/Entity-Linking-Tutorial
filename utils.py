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