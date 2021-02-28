import argparse
import sys, json
from distutils.util import strtobool

class Biencoder_params:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Entity linker')
        parser.add_argument('-debug', action='store', default=False, type=strtobool)
        parser.add_argument('-debug_for_entity_encoder', action='store', default=False, type=strtobool)
        parser.add_argument('-debug_with_onlygoldentities', action='store', default=False, type=strtobool)
        parser.add_argument('-dataset', action="store", default="med", dest="dataset", type=str)

        parser.add_argument('-cached_instance', action='store', default=False, type=strtobool)
        parser.add_argument('-lr', action="store", default=1e-5, type=float)
        parser.add_argument('-weight_decay', action="store", default=0, type=float)
        parser.add_argument('-beta1', action="store", default=0.9, type=float)
        parser.add_argument('-beta2', action="store", default=0.999, type=float)
        parser.add_argument('-epsilon', action="store", default=1e-8, type=float)
        parser.add_argument('-amsgrad', action='store', default=False, type=strtobool)
        parser.add_argument('-word_embedding_dropout', action="store", default=0.1, type=float)
        parser.add_argument('-scalingMSEfactor', action="store", default=1.0, type=float)
        parser.add_argument('-save_model', action="store", default=1, type=int)
        parser.add_argument('-cuda_devices', action="store", default='0', type=str)

        parser.add_argument('-num_epochs', action="store", default=30, type=int)
        parser.add_argument('-batch_size_for_train', action="store", default=32, type=int)
        parser.add_argument('-batch_size_for_eval', action="store", default=32, type=int)
        parser.add_argument('-hard_negatives_num', action="store", default=10, type=int)
        parser.add_argument('-train_data_num_sampled', action="store", default=-1, type=int) # -1: use all


        parser.add_argument('-allen_lazyload', action='store', default=True, type=strtobool)
        parser.add_argument('-add_hard_negatives', action='store', default=False, type=strtobool)
        parser.add_argument('-bert_name', action='store', default='bert-base-uncased', type=str)

        # For deciding limits of maximum token length
        parser.add_argument('-max_context_len', action="store", default=80, type=int)
        parser.add_argument('-max_mention_len', action="store", default=12, type=int)
        parser.add_argument('-max_left_context_len', action="store", default=35, type=int)
        parser.add_argument('-max_right_context_len', action="store", default=35, type=int)
        parser.add_argument('-max_canonical_len', action="store", default=12, type=int)
        parser.add_argument('-max_def_len', action="store", default=48, type=int)

        # Filepaths for fixed data
        parser.add_argument('-experiment_logdir', action='store', default='./experiment_logdir/', type=str)
        parser.add_argument('-mention_dump_dir', action='store', default='./mention_dump_dir/', type=str)

        # for Loading/Constructing KB
        parser.add_argument('-kbemb_dim', action="store", default=300, type=int)
        parser.add_argument('-negatives_for_knn', action="store", default=500, type=int)
        parser.add_argument('-cand_num_for_knn', action="store", default=500, type=int)

        # train_kg_or_biencoder
        parser.add_argument('-model_for_training', action="store", default='blink', type=str) # [kgann, biencoder]

        # For BLINKBiencoder
        parser.add_argument('-cand_num_before_sort_candidates_forBLINKbiencoder', action="store", default=500, type=int)
        parser.add_argument('-searchMethodWithFaiss', action='store', default='indexflatip', type=str)

        self.opts = parser.parse_args(sys.argv[1:])
        print('\n===PARAMETERS===')
        for arg in vars(self.opts):
            print(arg, getattr(self.opts, arg))
        print('===PARAMETERS END===\n')

        # Enforced params
        self.allen_lazyloader()
        self.set_maximum_textlenght_foreachdataset()

    def get_params(self):
        return self.opts

    def allen_lazyloader(self):
        if self.opts.dataset in ['st21pv', 'med']:
            self.opts.allen_lazyload = True

    def set_maximum_textlenght_foreachdataset(self):
        if self.opts.dataset in ['st21pv', 'med']:
            self.opts.max_left_context_len = 30
            self.opts.max_right_context_len = 30
            self.opts.max_def_len = 40

    def dump_params(self, experiment_dir):
        parameters = vars(self.get_params())
        with open(experiment_dir + 'parameters.json', 'w') as f:
            json.dump(parameters, f, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))