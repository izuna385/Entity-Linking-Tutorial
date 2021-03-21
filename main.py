from dataset_reader import BC5CDRReader
from parameteres import Biencoder_params
from utils import build_vocab, build_data_loaders, build_one_flag_loader, emb_returner, build_trainer
from encoder import Pooler_for_mention, Pooler_for_cano_and_def
from model import Biencoder
from allennlp.training.util import evaluate
import copy
from evaluator import BiencoderSqueezedCandidateEvaluator

if __name__ == '__main__':
    config = Biencoder_params()
    params = config.opts
    reader = BC5CDRReader(params)

    # Loading Datasets
    train, dev, test = reader._read('train'), reader._read('dev'), reader._read('test')
    vocab = build_vocab(train)
    vocab.extend_from_instances(dev)

    train_loader, dev_loader, test_loader = build_data_loaders(params, train, dev, test)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    _, __, embedder = emb_returner(config=params)
    mention_encoder, entity_encoder = Pooler_for_mention(params, embedder), Pooler_for_cano_and_def(params, embedder)

    model = Biencoder(params, mention_encoder, entity_encoder, vocab)

    trainer = build_trainer(params, model, train_loader, dev_loader)
    trainer.train()

    # switch to evaluation model
    model.istrainflag = copy.copy(0)
    model.eval()

    # switch to dev evaluation mode
    reader.dev_eval_flag = copy.copy(1)
    dev = reader._read('dev')
    dev_loader = build_one_flag_loader(config, dev)
    dev_loader.index_with(vocab)
    test_loader.index_with(model.vocab)

    squeezed_evaluator_model = BiencoderSqueezedCandidateEvaluator(params, mention_encoder, entity_encoder, vocab)
    squeezed_evaluator_model.eval()
    dev_eval_result = evaluate(model=squeezed_evaluator_model, data_loader=dev_loader, cuda_device=0,
                               batch_weight_key="")
    print(dev_eval_result)
    test_eval_result = evaluate(model=squeezed_evaluator_model, data_loader=test_loader, cuda_device=0,
                                batch_weight_key="")
    print(test_eval_result)