import transformers
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
import os
from transformers import AutoTokenizer, AutoModel
import urllib.request
from parameteres import Biencoder_params
from commons import MENTION_START_TOKEN, MENTION_END_TOKEN

class CustomTokenizer:
    def __init__(self, config):
        self.config = config
        self.bert_model_and_vocab_downloader()
        self.bert_tokenizer = self.bert_tokenizer_returner()

    def huggingfacename_returner(self):
        'Return huggingface modelname and do_lower_case parameter'
        if self.config.bert_name == 'bert-base-uncased':
            return 'bert-base-uncased', True
        elif self.config.bert_name == 'biobert':
            # https://huggingface.co/monologg/biobert_v1.1_pubmed/tree/main
            return './biobert/', False
        else:
            print('Currently', self.config.bert_name, 'are not supported.')
            exit()

    def token_indexer_returner(self):
        huggingface_name, do_lower_case = self.huggingfacename_returner()
        return {'tokens': PretrainedTransformerIndexer(
            model_name=huggingface_name,
            # do_lowercase=do_lower_case
        )
        }

    def bert_tokenizer_returner(self):
        if self.config.bert_name == 'bert-base-uncased':
            vocab_file = './vocab_file/bert-base-uncased-vocab.txt'
            do_lower_case = True
            return transformers.BertTokenizer(vocab_file=vocab_file,
                                              do_lower_case=do_lower_case,
                                              do_basic_tokenize=True,
                                              never_split=['<target>', '</target>'])
        elif self.config.bert_name == 'biobert':
            vocab_file = './vocab_file/biobert_v1.1_pubmed_vocab.txt'
            do_lower_case = False
            return transformers.BertTokenizer(vocab_file=vocab_file,
                                              do_lower_case=do_lower_case,
                                              do_basic_tokenize=True,
                                              never_split=['<target>', '</target>'])
        else:
            print('currently not supported:', self.config.bert_name)
            raise NotImplementedError


    def tokenize(self, txt):
        target_anchors = ['<target>', '</target>']
        original_tokens = txt.split(' ')
        new_tokens = list()

        for token in original_tokens:
            if token in target_anchors:
                if token == '<target>':
                    new_tokens.append(MENTION_START_TOKEN)
                if token == '</target>':
                    new_tokens.append(MENTION_END_TOKEN)
                continue
            else:
                split_to_subwords = self.bert_tokenizer.tokenize(token)  # token is oneword, split_tokens
                if ['[CLS]'] in split_to_subwords:
                    split_to_subwords.remove('[CLS]')
                if ['[SEP]'] in split_to_subwords:
                    split_to_subwords.remove('[SEP]')
                if split_to_subwords == []:
                    new_tokens.append('[UNK]')
                else:
                    new_tokens += split_to_subwords

        return new_tokens

    def bert_model_and_vocab_downloader(self):
        if self.config.bert_name == 'biobert':
            if not os.path.exists('./biobert/'):
                os.mkdir('./biobert/')
                print('=== Downloading biobert ===')
                urllib.request.urlretrieve("https://huggingface.co/monologg/biobert_v1.0_pubmed_pmc/blob/main/config.json", './biobert/config.json')
                urllib.request.urlretrieve("https://huggingface.co/monologg/biobert_v1.0_pubmed_pmc/blob/main/pytorch_model.bin", './biobert/pytorch_model.bin')
                urllib.request.urlretrieve("https://huggingface.co/monologg/biobert_v1.0_pubmed_pmc/blob/main/special_tokens_map.json", './biobert/special_tokens_map.json')
                urllib.request.urlretrieve("https://huggingface.co/monologg/biobert_v1.0_pubmed_pmc/blob/main/tokenizer_config.json", './biobert/tokenizer_config.json')

        if not os.path.exists('./vocab_file/'):
            os.mkdir('./vocab_file/')

        bert_base_uncased_vocab_url = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
        bibobert_vocab_url = "https://huggingface.co/monologg/biobert_v1.1_pubmed/blob/main/vocab.txt"

        urllib.request.urlretrieve(bert_base_uncased_vocab_url, './vocab_file/bert-base-uncased-vocab.txt')
        urllib.request.urlretrieve(bibobert_vocab_url, './vocab_file/biobert_v1.1_pubmed_vocab.txt')


if __name__ == '__main__':
    config = Biencoder_params()
    params = config.opts
    tokenizer = CustomTokenizer(config=params)