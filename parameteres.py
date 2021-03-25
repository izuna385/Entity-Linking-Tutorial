import argparse
import sys, json
from distutils.util import strtobool

class Biencoder_params:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Entity linker')
        parser.add_argument('-debug', action='store', default=False, type=strtobool)
        parser.add_argument('-debug_data_num', action='store', default=200, type=int)
        parser.add_argument('-dataset', action="store", default="bc5cdr", dest="dataset", type=str)
        parser.add_argument('-dataset_dir', action="store", default="./dataset/", type=str)
        parser.add_argument('-serialization_dir', action="store", default="./serialization_dir/", type=str)
        parser.add_argument('-preprocessed_doc_dir', action="store", default="./preprocessed_doc_dir/", type=str)
        parser.add_argument('-kb_dir', action="store", default="./mesh/", type=str)

        parser.add_argument('-cached_instance', action='store', default=False, type=strtobool)
        parser.add_argument('-lr', action="store", default=1e-5, type=float)
        parser.add_argument('-weight_decay', action="store", default=0, type=float)
        parser.add_argument('-beta1', action="store", default=0.9, type=float)
        parser.add_argument('-beta2', action="store", default=0.999, type=float)
        parser.add_argument('-epsilon', action="store", default=1e-8, type=float)
        parser.add_argument('-amsgrad', action='store', default=False, type=strtobool)
        parser.add_argument('-word_embedding_dropout', action="store", default=0.1, type=float)
        parser.add_argument('-cuda_devices', action="store", default='0', type=str)
        parser.add_argument('-scoring_function_for_model', action="store", default='indexflatip', type=str)

        parser.add_argument('-num_epochs', action="store", default=10, type=int)
        parser.add_argument('-patience', action="store", default=10, type=int)
        parser.add_argument('-batch_size_for_train', action="store", default=16, type=int)
        parser.add_argument('-batch_size_for_eval', action="store", default=16, type=int)

        parser.add_argument('-bert_name', action='store', default='bert-base-uncased', type=str)

        # For deciding limits of maximum token length
        parser.add_argument('-max_context_len', action="store", default=50, type=int)
        parser.add_argument('-max_mention_len', action="store", default=12, type=int)
        parser.add_argument('-max_canonical_len', action="store", default=12, type=int)
        parser.add_argument('-max_def_len', action="store", default=36, type=int)

        # train_kg_or_biencoder
        parser.add_argument('-model_for_training', action="store", default='blink', type=str) # [kgann, biencoder]

        parser.add_argument('-candidates_dataset', action='store', default='./candidates.pkl', type=str)
        parser.add_argument('-max_candidates_num', action='store', default=10, type=int)

        # for entire kb eval.
        parser.add_argument('-search_method_for_faiss', action='store', default='indexflatip', type=str)

        # Note: Currently we do not support other candidate numbers. See evaluate_with_entire_kb.py.
        parser.add_argument('-how_many_top_hits_preserved', action='store', default=50, type=int)

        self.opts = parser.parse_args(sys.argv[1:])
        print('\n===PARAMETERS===')
        for arg in vars(self.opts):
            print(arg, getattr(self.opts, arg))
        print('===PARAMETERS END===\n')

    def get_params(self):
        return self.opts

    def dump_params(self, experiment_dir):
        parameters = vars(self.get_params())
        with open(experiment_dir + 'parameters.json', 'w') as f:
            json.dump(parameters, f, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))