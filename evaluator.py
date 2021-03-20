import pdb
from allennlp.nn import util as nn_util
from tqdm import tqdm
import torch
from torch.nn.functional import normalize
import numpy as np
import math, json

import torch
import torch.nn as nn
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.models import Model
from overrides import overrides
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy
from torch.nn.functional import normalize
import torch.nn.functional as F

class BiencoderEvaluator(Model):
    def __init__(self, args,
                 mention_encoder: Seq2VecEncoder,
                 entity_encoder: Seq2VecEncoder,
                 vocab):
        super().__init__(vocab)
        self.args = args
        self.mention_encoder = mention_encoder
        self.accuracy = CategoricalAccuracy()
        self.BCEWloss = nn.BCEWithLogitsLoss()
        self.mesloss = nn.MSELoss()
        self.entity_encoder = entity_encoder

    def forward(self, context, gold_dui_canonical_and_def_concatenated, gold_duidx, mention_uniq_id,
                candidates_canonical_and_def_concatenated, gold_location_in_candidates):
        batch_num = context['tokens']['token_ids'].size(0)
        device = torch.get_device(context['tokens']['token_ids']) if torch.cuda.is_available() else torch.device('cpu')
        contextualized_mention = self.mention_encoder(context)
        encoded_entites = self._candidate_entities_emb_returner(batch_num, candidates_canonical_and_def_concatenated)

        if self.args.scoring_function_for_model == 'cossim':
            contextualized_mention_forcossim = normalize(contextualized_mention, dim=1)
            encoded_entites_forcossim = normalize(encoded_entites, dim=2)
            scores = torch.bmm(encoded_entites_forcossim.unsqueeze(2), contextualized_mention_forcossim.view(batch_num, -1, 1)).squeeze()
        elif self.args.scoring_function_for_model == 'indexflatip':
            scores = torch.bmm(encoded_entites, contextualized_mention.view(batch_num, -1, 1)).squeeze()
        else:
            assert self.args.searchMethodWithFaiss == 'indexflatl2'
            raise NotImplementedError

        loss =  self.BCEWloss(scores, gold_location_in_candidates.squeeze(1).float())
        output = {'loss': loss}
        self.accuracy(scores, torch.argmax(gold_location_in_candidates.squeeze(1), dim=1))

        return output

    @overrides
    def get_metrics(self, reset: bool = False):
        return {"accuracy": self.accuracy.get_metric(reset)}

    def return_entity_encoder(self):
        return self.entity_encoder

    def _candidate_entities_emb_returner(self, batch_size, candidates_canonical_and_def_concatenated):
        cand_nums = candidates_canonical_and_def_concatenated['tokens']['token_ids'].size(1)

        candidates_canonical_and_def_concatenated['tokens']['token_ids'] = \
            candidates_canonical_and_def_concatenated['tokens']['token_ids'].view(batch_size * cand_nums, -1)

        candidates_canonical_and_def_concatenated['tokens']['mask'] = \
            candidates_canonical_and_def_concatenated['tokens']['mask'].view(batch_size * cand_nums, -1)

        candidates_canonical_and_def_concatenated['tokens']['type_ids'] = \
            candidates_canonical_and_def_concatenated['tokens']['type_ids'].view(batch_size * cand_nums, -1)

        candidate_embs = self.entity_encoder(candidates_canonical_and_def_concatenated)
        candidate_embs = candidate_embs.view(batch_size, cand_nums, -1)

        return candidate_embs