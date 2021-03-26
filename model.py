'''
Model classes
'''
import torch
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.models import Model
from overrides import overrides
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy
from torch.nn.functional import normalize
import torch.nn.functional as F
import torch.nn as nn
import pdb

class Biencoder(Model):
    def __init__(self, args,
                 mention_encoder: Seq2VecEncoder,
                 entity_encoder: Seq2VecEncoder,
                 vocab):
        super().__init__(vocab)
        self.args = args
        self.mention_encoder = mention_encoder
        self.accuracy = CategoricalAccuracy()
        self.entity_encoder = entity_encoder

        self.istrainflag = 1

    def forward(self, gold_dui_canonical_and_def_concatenated,
                context: torch.Tensor = None,
                gold_duidx: torch.Tensor = None,
                mention_uniq_id: torch.Tensor = None,
                candidates_canonical_and_def_concatenated: torch.Tensor = None,
                gold_location_in_candidates: torch.Tensor = None):
        if context == None and gold_duidx == None and mention_uniq_id == None and \
                candidates_canonical_and_def_concatenated == None and gold_location_in_candidates == None:
            output = {}
            output['encoded_entities'] = self.entity_encoder(
                cano_and_def_concatnated_text=gold_dui_canonical_and_def_concatenated)
            return output

        batch_num = context['tokens']['token_ids'].size(0)
        device = torch.get_device(context['tokens']['token_ids']) if torch.cuda.is_available() else torch.device('cpu')
        contextualized_mention = self.mention_encoder(context)
        encoded_entites = self.entity_encoder(cano_and_def_concatnated_text=gold_dui_canonical_and_def_concatenated)

        if self.args.scoring_function_for_model == 'cossim':
            contextualized_mention = normalize(contextualized_mention, dim=1)
            encoded_entites = normalize(encoded_entites, dim=1)

        encoded_entites = encoded_entites.squeeze(1)
        dot_product = torch.matmul(contextualized_mention, encoded_entites.t())  # [bs, bs]
        mask = torch.eye(batch_num).to(device)
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()

        output = {'loss': loss}
        if self.istrainflag:
            golds = torch.eye(batch_num).to(device)
            self.accuracy(dot_product, torch.argmax(golds, dim=1))

        else:
            output['gold_duidx'] = gold_duidx
            output['encoded_mentions'] = contextualized_mention

        return output

    @overrides
    def get_metrics(self, reset: bool = False):
        return {"accuracy": self.accuracy.get_metric(reset)}

    def return_entity_encoder(self):
        return self.entity_encoder

class BiencoderSqueezedCandidateEvaluator(Model):
    '''
    For dev and test evaluation with Surface-based candidates.
    '''
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
            scores = torch.bmm(encoded_entites_forcossim, contextualized_mention_forcossim.view(batch_num, -1, 1)).squeeze()
        elif self.args.scoring_function_for_model == 'indexflatip':
            scores = torch.bmm(encoded_entites, contextualized_mention.view(batch_num, -1, 1)).squeeze()
        else:
            assert self.args.searchMethodWithFaiss == 'indexflatl2'
            raise NotImplementedError

        loss = self.BCEWloss(scores, gold_location_in_candidates.view(batch_num, -1).float())
        output = {'loss': loss,
                  'contextualized_mention': contextualized_mention}
        self.accuracy(scores, torch.argmax(gold_location_in_candidates.view(batch_num, -1), dim=1))

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

class BiencoderNNSearchEvaluator(Model):
    '''
    For dev and test evaluation with Approximante Nearest Neighbor Search based candidates.
    '''
    def __init__(self, args,
                 mention_encoder: Seq2VecEncoder,
                 vocab, faiss_searcher):
        super().__init__(vocab)
        self.args = args
        self.mention_encoder = mention_encoder
        self.accuracy = CategoricalAccuracy()
        self.faiss_searcher = faiss_searcher
        self.mention_idx2candidate_entity_idxs = {}

    def forward(self, context, gold_dui_canonical_and_def_concatenated, gold_duidx, mention_uniq_id,
                candidates_canonical_and_def_concatenated, gold_location_in_candidates):
        contextualized_mention = self.mention_encoder(context)
        distances, in_faiss_idxes = self.faiss_searcher.indexed_faiss.search(contextualized_mention.cpu().numpy(),
                                                                             k=self.args.how_many_top_hits_preserved)
        for mention_idx, in_faiss_candidates, gold_duidx_ in zip(mention_uniq_id.cpu().numpy(), in_faiss_idxes, gold_duidx.cpu().numpy()):
            candidate_entity_idxes = [self.faiss_searcher.kb_idx2entity_idx[idx]
                                                                        for idx in in_faiss_candidates]
            self.mention_idx2candidate_entity_idxs.update({mention_idx:
                                                              {'candidate_entity_idx':candidate_entity_idxes,
                                                               'gold_entity_idx': gold_duidx_}})
        output = {}

        return output

    @overrides
    def get_metrics(self, reset: bool = False):
        return {"accuracy": self.accuracy.get_metric(reset)}