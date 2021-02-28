from dataset_reader import BC5CDRReader
from parameteres import Biencoder_params
from utils import build_vocab, build_data_loaders, emb_returner, build_trainer
from encoder import Pooler_for_mention, Pooler_for_cano_and_def
from model import Biencoder
import pdb

if __name__ == '__main__':
    config = Biencoder_params()
    params = config.opts
    reader = BC5CDRReader(params)

    # Loading Datasets
    train, dev, test, train_and_dev = reader._read('train'), reader._read('dev'), reader._read('test'), \
                                      reader._read('train_and_dev')

    vocab = build_vocab(train_and_dev)

    train_loader, dev_loader = build_data_loaders(params, train, dev)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    _, __, embedder = emb_returner(config=params)
    mention_encoder, entity_encoder = Pooler_for_mention(params, embedder), Pooler_for_cano_and_def(params, embedder)

    model = Biencoder(params, mention_encoder,entity_encoder, vocab)

    trainer = build_trainer(params, model, train_loader, dev_loader)
    trainer.train()