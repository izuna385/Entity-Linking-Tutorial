import tempfile
from typing import Dict, Iterable
from overrides import overrides
from commons import CANONICAL_AND_DEF_CONNECTTOKEN, MENTION_START_TOKEN, MENTION_END_TOKEN
import torch
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import SpanField, ListField, TextField, MetadataField, ArrayField, SequenceLabelField, LabelField
from allennlp.data.fields import LabelField, TextField
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from parameteres import Biencoder_params
import glob
import os
import random
import pdb
from tqdm import tqdm
import json
from tokenizer import CustomTokenizer
import numpy as np
from candidate_generator import CandidateGeneratorForTestDataset

class BC5CDRReader(DatasetReader):
    def __init__(
        self,
        config,
        max_tokens: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.custom_tokenizer_class = CustomTokenizer(config=config)
        self.token_indexers = self.custom_tokenizer_class.token_indexer_returner()
        self.max_tokens = max_tokens
        self.config = config
        self.train_pmids, self.dev_pmids, self.test_pmids = self._train_dev_test_pmid_returner()
        self.id2mention, self.train_mention_ids, self.dev_mention_ids, self.test_mention_ids = \
            self._mention_id_returner(self.train_pmids, self.dev_pmids, self.test_pmids)

        # kb loading
        self.dui2idx, self.idx2dui, self.dui2canonical, self.dui2definition = self._kb_loader()
        self.candidate_generator = CandidateGeneratorForTestDataset(config=config)
        self.dev_eval_flag = 0

        self.dev_recall, self.test_recall = 0, 0

    @overrides
    def _read(self, train_dev_test_flag: str) -> list:
        '''
        :param train_dev_test_flag: 'train', 'dev', 'test'
        :return: list of instances
        '''
        mention_ids, instances = list(), list()
        if train_dev_test_flag == 'train':
            mention_ids += self.train_mention_ids
            # Because Iterator(shuffle=True) has bug, we forcefully shuffle train dataset here.
            random.shuffle(mention_ids)
        elif train_dev_test_flag == 'dev':
            mention_ids += self.dev_mention_ids
        elif train_dev_test_flag == 'test':
            mention_ids += self.test_mention_ids
        elif train_dev_test_flag == 'train_and_dev':
            mention_ids += self.train_mention_ids
            mention_ids += self.dev_mention_ids

        if self.config.debug:
            mention_ids = mention_ids[:self.config.debug_data_num]

        ignored_mentions_num = 0
        for idx, mention_uniq_id in tqdm(enumerate(mention_ids)):
            # try:
            data = self._one_line_parser(mention_uniq_id=mention_uniq_id,
                                         train_dev_test_flag=train_dev_test_flag)
            if data['gold_duidx'] == -1:
                # print(mention_uniq_id, self.id2mention[mention_uniq_id])
                # print('Warning. This CUI is not included in MeSH Canonical and Definition Dictionary.')
                ignored_mentions_num += 1
                continue
            instances.append(self.text_to_instance(data=data))
                # yield self.text_to_instance(data=data)
            # except:
            #     continue
        print(train_dev_test_flag, 'ignored_mentions:', ignored_mentions_num)
        print('These mentions are ignored because the latest version\'s MeSH is used for indexing.')

        return instances


    def _train_dev_test_pmid_returner(self):
        '''
        :return: pmids list for using and evaluating entity linking task
        '''
        train_pmids, dev_pmids, test_pmids = self._pmid_returner('train'), self._pmid_returner('dev'), \
                                             self._pmid_returner('test')
        train_pmids = [pmid for pmid in train_pmids if self._is_parsed_doc_exist_per_pmid(pmid)]
        dev_pmids = [pmid for pmid in dev_pmids if self._is_parsed_doc_exist_per_pmid(pmid)]
        test_pmids = [pmid for pmid in test_pmids if self._is_parsed_doc_exist_per_pmid(pmid)]

        return train_pmids, dev_pmids, test_pmids

    def _pmid_returner(self, train_dev_test_flag: str):
        '''
        :param train_dev_test_flag: train, dev, test
        :return: pmids (str list)
        '''
        assert train_dev_test_flag in ['train', 'dev', 'test']
        pmid_dir = self.config.dataset_dir
        pmids_txt_path = pmid_dir + 'corpus_pubtator_pmids_'
        if train_dev_test_flag == 'train':
            pmids_txt_path += 'trng'
        else:
            pmids_txt_path += train_dev_test_flag
        pmids_txt_path += '.txt'

        pmids = []
        with open(pmids_txt_path, 'r') as p:
            for line in p:
                line = line.strip()
                if line != '':
                    pmids.append(line)

        return pmids

    def _is_parsed_doc_exist_per_pmid(self, pmid: str):
        '''
        :param pmid:
        :return: if parsed doc exists in ./preprocessed_doc_dir/
        '''
        if os.path.exists(self.config.preprocessed_doc_dir + pmid + '.json'):
            return 1
        else:
            return 0

    def _mention_id_returner(self, train_pmids: list, dev_pmids: list, test_pmids: list):
        id2mention, train_mention_ids, dev_mention_ids, test_mention_ids = {}, [], [], []
        for pmid in train_pmids:
            mentions = self._pmid2mentions(pmid)
            for mention in mentions:
                id = len(id2mention)
                id2mention.update({id: mention})
                train_mention_ids.append(id)

        for pmid in dev_pmids:
            mentions = self._pmid2mentions(pmid)
            for mention in mentions:
                id = len(id2mention)
                id2mention.update({id: mention})
                dev_mention_ids.append(id)

        for pmid in test_pmids:
            mentions = self._pmid2mentions(pmid)
            for mention in mentions:
                id = len(id2mention)
                id2mention.update({id: mention})
                test_mention_ids.append(id)

        return id2mention, train_mention_ids, dev_mention_ids, test_mention_ids

    def _pmid2mentions(self, pmid):
        parsed_doc_json_path = self.config.preprocessed_doc_dir + pmid + '.json'
        with open(parsed_doc_json_path, 'r') as pd:
            parsed = json.load(pd)
        mentions = parsed['lines']

        return mentions

    def _kb_loader(self):
        kb_dir = self.config.kb_dir
        with open(kb_dir + 'dui2canonical.json', 'r') as f:
            dui2canonical = json.load(f)

        with open(kb_dir + 'dui2definition.json', 'r') as g:
            dui2definition = json.load(g)

        with open(kb_dir + 'dui2idx.json', 'r') as h:
            dui2idx_ = json.load(h)
        dui2idx = {}
        for dui, idx_str in dui2idx_.items():
            dui2idx.update({dui: int(idx_str)})

        with open(kb_dir + 'idx2dui.json', 'r') as k:
            idx2dui_ = json.load(k)
        idx2dui = {}
        for idx_str, dui in idx2dui_.items():
            idx2dui.update({int(idx_str): dui})

        return dui2idx, idx2dui, dui2canonical, dui2definition

    def _one_line_parser(self, mention_uniq_id, train_dev_test_flag='train'):
        if train_dev_test_flag in ['train'] or (train_dev_test_flag == 'dev' and self.dev_eval_flag == 0):
            line = self.id2mention[mention_uniq_id]
            gold_dui, _, gold_surface_mention, target_anchor_included_sentence = line.split('\t')
            tokenized_context_including_target_anchors = self.custom_tokenizer_class.tokenize(
                txt=target_anchor_included_sentence)
            tokenized_context_including_target_anchors = [Token(split_token) for split_token in
                                                          tokenized_context_including_target_anchors]
            data = {'context': tokenized_context_including_target_anchors}

            data['mention_uniq_id'] = int(mention_uniq_id)
            data['gold_duidx'] = int(self.dui2idx[gold_dui]) if gold_dui in self.dui2idx and gold_dui in self.dui2canonical else -1
            if gold_dui in self.dui2canonical:
                data['gold_dui_canonical_and_def_concatenated'] = self._canonical_and_def_context_concatenator(dui=gold_dui)
        else:
            assert train_dev_test_flag in ['dev', 'test']
            line = self.id2mention[mention_uniq_id]
            gold_dui, _, surface_mention, target_anchor_included_sentence = line.split('\t')

            candidate_duis_idx = [self.dui2idx[dui] for dui in self.candidate_generator.mention2candidate_duis[surface_mention]
                              if dui in self.dui2idx and dui in self.dui2canonical][:self.config.max_candidates_num]
            while len(candidate_duis_idx) < self.config.max_candidates_num:
                random_choiced_dui = random.choice([dui for dui in self.dui2idx.keys()])
                if self.dui2idx[random_choiced_dui] not in candidate_duis_idx:
                    candidate_duis_idx.append(self.dui2idx[random_choiced_dui])
            tokenized_context_including_target_anchors = self.custom_tokenizer_class.tokenize(
                txt=target_anchor_included_sentence)
            tokenized_context_including_target_anchors = [Token(split_token) for split_token in
                                                          tokenized_context_including_target_anchors][:self.config.max_context_len]
            data = {'context': tokenized_context_including_target_anchors}
            data['candidate_duis_idx'] = candidate_duis_idx
            data['gold_duidx'] = int(self.dui2idx[gold_dui]) if gold_dui in self.dui2idx else -1

            gold_location_in_candidates = [0 for _ in range(self.config.max_candidates_num)]

            if gold_dui in self.dui2idx:
                for idx, cand_idx in enumerate(candidate_duis_idx):
                    if cand_idx == self.dui2idx[gold_dui]:
                        gold_location_in_candidates[idx] += 1

                        if train_dev_test_flag == 'dev':
                            self.dev_recall += 1
                        if train_dev_test_flag == 'test':
                            self.test_recall += 1

            data['gold_location_in_candidates'] = gold_location_in_candidates
            data['mention_uniq_id'] = int(mention_uniq_id)

        return data

    def _canonical_and_def_context_concatenator(self, dui):
        canonical =  self.custom_tokenizer_class.tokenize(txt=self.dui2canonical[dui])
        definition =  self.custom_tokenizer_class.tokenize(txt=self.dui2definition[dui])
        concatenated = ['[CLS]']
        concatenated += canonical[:self.config.max_canonical_len]
        concatenated.append(CANONICAL_AND_DEF_CONNECTTOKEN)
        concatenated += definition[:self.config.max_def_len]
        concatenated.append('[SEP]')

        return [Token(tokenized_word) for tokenized_word in concatenated]

    @overrides
    def text_to_instance(self, data=None) -> Instance:
        context_field = TextField(data['context'], self.token_indexers)
        fields = {"context": context_field}

        fields['gold_duidx'] = ArrayField(np.array(data['gold_duidx']))
        fields['mention_uniq_id'] = ArrayField(np.array(data['mention_uniq_id']))

        if data['mention_uniq_id'] in self.test_mention_ids or \
                (data['mention_uniq_id'] in self.dev_mention_ids and self.dev_eval_flag):
            candidates_canonical_and_def_concatenated = [TextField(self._canonical_and_def_context_concatenator(
                dui=self.idx2dui[idx]), self.token_indexers) for idx in data['candidate_duis_idx']]
            fields['candidates_canonical_and_def_concatenated'] = ListField(candidates_canonical_and_def_concatenated)
            fields['gold_location_in_candidates'] = ArrayField(np.array([data['gold_location_in_candidates']],
                                                                        dtype='int16'))
            fields['gold_dui_canonical_and_def_concatenated'] = MetadataField(0)
        else: # train, or dev-eval under train
            fields['candidates_canonical_and_def_concatenated'] = MetadataField(0)
            fields['gold_location_in_candidates'] = MetadataField(0)
            fields['gold_dui_canonical_and_def_concatenated'] = TextField(
                data['gold_dui_canonical_and_def_concatenated'],
                self.token_indexers)

        return Instance(fields)


'''
For iterating all entities
'''
class EntitiesInKBLoader(DatasetReader):
    def __init__(
            self,
            config,
            max_tokens: int = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.custom_tokenizer_class = CustomTokenizer(config=config)
        self.token_indexers = self.custom_tokenizer_class.token_indexer_returner()
        self.max_tokens = max_tokens
        self.config = config

        # kb loading
        self.dui2idx, self.idx2dui, self.dui2canonical, self.dui2definition = self._kb_loader()

    @overrides
    def _read(self) -> list:
        '''
        :param train_dev_test_flag: 'train', 'dev', 'test'
        :return: list of instances
        '''
        for entity_unique_id, dui in tqdm(enumerate(self.idx2dui.items())):
            if self.config.debug and entity_unique_id == 100:
                break
            instance = self.text_to_instance(entity_unique_id=entity_unique_id)
            yield instance

    def _one_entity_parser(self, entity_uniq_id: int):
        gold_dui = self.idx2dui[entity_uniq_id]
        data = {}
        data['entity_uniq_id'] = int(entity_uniq_id)
        data['gold_dui_canonical_and_def_concatenated'] = self._canonical_and_def_context_concatenator(
            dui=gold_dui)

        return data

    def _canonical_and_def_context_concatenator(self, dui):
        canonical = self.custom_tokenizer_class.tokenize(txt=self.dui2canonical[dui])
        definition = self.custom_tokenizer_class.tokenize(txt=self.dui2definition[dui])
        concatenated = ['[CLS]']
        concatenated += canonical[:self.config.max_canonical_len]
        concatenated.append(CANONICAL_AND_DEF_CONNECTTOKEN)
        concatenated += definition[:self.config.max_def_len]
        concatenated.append('[SEP]')

        return [Token(tokenized_word) for tokenized_word in concatenated]

    @overrides
    def text_to_instance(self, entity_unique_id=None) -> Instance:
        data = self._one_entity_parser(entity_uniq_id=entity_unique_id)
        fields = {}

        fields['gold_dui_canonical_and_def_concatenated'] = TextField(
            data['gold_dui_canonical_and_def_concatenated'], self.token_indexers)

        return Instance(fields)

    def _kb_loader(self):
        kb_dir = self.config.kb_dir
        with open(kb_dir + 'dui2canonical.json', 'r') as f:
            dui2canonical = json.load(f)

        with open(kb_dir + 'dui2definition.json', 'r') as g:
            dui2definition = json.load(g)

        with open(kb_dir + 'dui2idx.json', 'r') as h:
            dui2idx_ = json.load(h)
        dui2idx = {}
        for dui, idx_str in dui2idx_.items():
            dui2idx.update({dui: int(idx_str)})

        with open(kb_dir + 'idx2dui.json', 'r') as k:
            idx2dui_ = json.load(k)
        idx2dui = {}
        for idx_str, dui in idx2dui_.items():
            idx2dui.update({int(idx_str): dui})

        return dui2idx, idx2dui, dui2canonical, dui2definition

    def get_entity_ids(self):
        return [idx for idx in self.idx2dui.keys()]