'''
Model classes
'''
import torch
import torch.nn as nn
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.models import Model
from overrides import overrides
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy
from torch.nn.functional import normalize
import torch.nn.functional as F
import copy
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
        self.BCEWloss = nn.BCEWithLogitsLoss()
        self.mesloss = nn.MSELoss()
        self.entity_encoder = entity_encoder

        self.istrainflag = 1

    def forward(self, context, gold_dui_canonical_and_def_concatenated, gold_duidx, mention_uniq_id,
                candidates_canonical_and_def_concatenated, gold_location_in_candidates):
        batch_num =  context['tokens']['token_ids'].size(0)
        device = torch.get_device(context['tokens']['token_ids']) if torch.cuda.is_available() else torch.device('cpu')
        contextualized_mention = self.mention_encoder(context)
        encoded_entites = self.entity_encoder(cano_and_def_concatnated_text=gold_dui_canonical_and_def_concatenated)

        if self.args.scoring_function_for_model == 'cossim':
            contextualized_mention_forcossim = normalize(contextualized_mention, dim=1)
            encoded_entites_forcossim = normalize(encoded_entites, dim=1)
            scores = contextualized_mention_forcossim.mm(encoded_entites_forcossim.t())
        elif self.args.scoring_function_for_model == 'indexflatip':
            scores = contextualized_mention.mm(encoded_entites.t())
        else:
            assert self.args.searchMethodWithFaiss == 'indexflatl2'
            raise NotImplementedError

        target = torch.LongTensor(torch.arange(batch_num)).to(device)
        # loss = F.cross_entropy(scores, target, reduction="mean")
        loss = self.BCEWloss(scores, torch.eye(batch_num).to(device))
        output = {'loss': loss}

        if self.istrainflag:
            golds = torch.eye(batch_num).to(device)
            self.accuracy(scores, torch.argmax(golds, dim=1))
        else:
            output['gold_duidx'] = gold_duidx
            output['encoded_mentions'] = contextualized_mention
        return output

    @overrides
    def get_metrics(self, reset: bool = False):
        return {"accuracy": self.accuracy.get_metric(reset)}

    def return_entity_encoder(self):
        return self.entity_encoder

    def switch2eval(self):
        self.istrainflag = copy.copy(0)

    def calc_L2distance(self, h, t):
        diff = h - t
        return torch.norm(diff, dim=2)  # batch * cands

class BLINKBiencoder_OnlyforEncodingMentions(Model):
    def __init__(self, args,
                 mention_encoder: Seq2VecEncoder,
                 vocab):
        super().__init__(vocab)
        self.args = args
        self.mention_encoder = mention_encoder

    def forward(self, context, gold_cui_cano_and_def_concatenated, gold_cuidx, mention_uniq_id):
        contextualized_mention = self.mention_encoder(context)
        output = {'mention_uniq_id': mention_uniq_id,
                  'gold_duidx': gold_cuidx,
                  'contextualized_mention': contextualized_mention}

        return output

class WrappedModel_for_entityencoding(Model):
    def __init__(self, args,
                 entity_encoder,
                 vocab):
        super().__init__(vocab)
        self.args = args
        self.entity_encoder = entity_encoder

    def forward(self, dui_idx, cano_and_def_concatenated):
        encoded_entites = self.entity_encoder(cano_and_def_concatnated_text=cano_and_def_concatenated)
        output = {'dui_idx': dui_idx, 'emb_of_entities_encoded': encoded_entites}

        return output

    def return_entity_encoder(self):
        return self.entity_encoder