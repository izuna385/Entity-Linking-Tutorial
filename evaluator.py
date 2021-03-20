import pdb
from allennlp.nn import util as nn_util
from allennlp.data.iterators import BasicIterator
from tqdm import tqdm
import torch
from torch.nn.functional import normalize
import numpy as np
import math, json

class BiencoderCandidateEvaluator:
    def __init__(self, config):
        self.config = config