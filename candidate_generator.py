import json, pickle, pdb
from parameteres import Biencoder_params

class CandidateGeneratorForTestDataset:
    def __init__(self, config):
        self.config = config
        self.mention2candidate_duis = self._dui2candidate_duis_returner()

    def _dui2candidate_duis_returner(self):
        with open(self.config.candidates_dataset, 'rb') as f:
            c = pickle.load(f)

        mention2candidate_duis = {}
        for mention, its_candidates in zip(c['mentions'], c['candidates']):
            mention2candidate_duis.update({mention: [dui for (dui, prior) in its_candidates]})

        return mention2candidate_duis

if __name__ == '__main__':
    config = Biencoder_params()
    params = config.opts
    cg = CandidateGeneratorForTestDataset(config=params)
    cg._dui2candidate_duis_returner()