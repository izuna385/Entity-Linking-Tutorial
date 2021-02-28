import pdb
from allennlp.nn import util as nn_util
from allennlp.data.iterators import BasicIterator
from tqdm import tqdm
import torch
from torch.nn.functional import normalize
import numpy as np
import math, json

class BiEncoderTopXRetriever:
    def __init__(self, args, vocab, mention_encoder, faiss_mesh, test):
        self.args = args
        self.mention_encoder = mention_encoder # BLINKBiencoder_OnlyforEncodingMentions
        self.mention_encoder.eval()
        self.faiss_searcher = faiss_mesh
        self.reader_for_mentions = test
        self.cuda_device = 0

    def blinkbiencoder_tophits_retrievaler(self, dev_or_test_flag, how_many_top_hits_preserved=100):
        assert dev_or_test_flag in ['train', 'dev', 'test']
        ds = self.reader_for_mentions.read(dev_or_test_flag)
        generator_for_biencoder = self.sequence_iterator(ds, num_epochs=1, shuffle=False)
        generator_for_biencoder_tqdm = tqdm(generator_for_biencoder, total=self.sequence_iterator.get_num_batches(ds))

        with torch.no_grad():
            for batch in generator_for_biencoder_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                mention_uniq_ids, encoded_mentions, gold_cuidxs = self._extract_mention_idx_encoded_emb_and_its_gold_cuidx(batch=batch)
                faiss_search_candidate_result_cuidxs = self.faiss_topx_retriever(encoded_mentions=encoded_mentions,
                                                                                 how_many_top_hits_preserved=how_many_top_hits_preserved)
                yield faiss_search_candidate_result_cuidxs, mention_uniq_ids, gold_cuidxs

    def faiss_topx_retriever(self, encoded_mentions, how_many_top_hits_preserved):
        '''
        if cossimsearch -> re-sort with L2, we have to use self.args.cand_num_before_sort_candidates_forBLINKbiencoder
        Args:
            encoded_mentions:
            how_many_top_hits_preserved:
        Returns:
        '''

        if self.args.searchMethodWithFaiss == 'cossim':
            encoded_mentions = normalize(torch.from_numpy(encoded_mentions), dim=1).cpu().detach().numpy()
            _, faiss_search_candidate_result_cuidxs = self.faiss_searcher.search(encoded_mentions, how_many_top_hits_preserved)

        else:
            # assert self.args.search_method_before_re_sorting_for_faiss == 'indexflatl2'
            _, faiss_search_candidate_result_cuidxs = self.faiss_searcher.search(encoded_mentions, how_many_top_hits_preserved)

        return faiss_search_candidate_result_cuidxs

    def calc_L2distance(self, h, t):
        diff = h - t
        return torch.norm(diff, dim=2)

    def tonp(self, tsr):
        return tsr.detach().cpu().numpy()

    def _extract_mention_idx_encoded_emb_and_its_gold_cuidx(self, batch) -> np.ndarray:
        out_dict = self.mention_encoder(**batch)
        return self.tonp(out_dict['mention_uniq_id']), self.tonp(out_dict['contextualized_mention']), self.tonp(out_dict['gold_cuidx'])

class DevandTest_BLINKBiEncoder_IterateEvaluator:
    def __init__(self, args, blinkBiEncoderEvaluator, experiment_logdir):
        self.blinkBiEncoderEvaluator = blinkBiEncoderEvaluator
        self.experiment_logdir = experiment_logdir

    def final_evaluation(self, dev_or_test_flag, how_many_top_hits_preserved=100):
        print('============\n<<<FINAL EVALUATION STARTS>>>', dev_or_test_flag, 'Retrieve_Candidates:', how_many_top_hits_preserved,'\n============')
        Hits1, Hits10, Hits50, Hits100 = 0, 0, 0, 0
        data_points = 0

        for faiss_search_candidate_result_cuidxs, mention_uniq_ids, gold_cuidxs in self.blinkBiEncoderEvaluator.blinkbiencoder_tophits_retrievaler(dev_or_test_flag, how_many_top_hits_preserved):
            b_Hits1, b_Hits10, b_Hits50, b_Hits100 = self.batch_candidates_and_gold_cuiddx_2_batch_hits(faiss_search_candidate_result_cuidxs=faiss_search_candidate_result_cuidxs,
                                                                                                        gold_cuidxs=gold_cuidxs)
            assert len(mention_uniq_ids) == len(gold_cuidxs)
            data_points += len(mention_uniq_ids)
            Hits1 += b_Hits1
            Hits10 += b_Hits10
            Hits50 += b_Hits50
            Hits100 += b_Hits100
        self.result_writer(dev_or_test_flag, data_points, Hits1, Hits10, Hits50, Hits100)

    def result_writer(self, dev_or_test_flag, data_points, Hits1, Hits10, Hits50, Hits100):
        result = {'dev_or_test': dev_or_test_flag,
                  'data_points': data_points,

                  'Hits1': str(Hits1 / data_points * 100) + ' %',
                  'Hits10': str(Hits10 / data_points * 100) + ' %',
                  'Hits50': str(Hits50 / data_points * 100) + ' %',
                  'Hits100': str(Hits100 / data_points * 100) + ' %'}
        with open(self.experiment_logdir + dev_or_test_flag + '_result.json', 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))
        if dev_or_test_flag == 'test':
            print('result_dir: ', self.experiment_logdir)

    def batch_candidates_and_gold_cuiddx_2_batch_hits(self, faiss_search_candidate_result_cuidxs, gold_cuidxs):
        b_Hits1, b_Hits10, b_Hits50, b_Hits100 = 0, 0, 0, 0
        for candidates_sorted, gold_idx in zip(faiss_search_candidate_result_cuidxs, gold_cuidxs):
            if len(np.where(candidates_sorted == int(gold_idx))[0]) != 0:
                rank = int(np.where(candidates_sorted == int(gold_idx))[0][0])

                if rank == 0:
                    b_Hits1 += 1
                    b_Hits10 += 1
                    b_Hits50 += 1
                    b_Hits100 += 1
                elif rank < 10:
                    b_Hits10 += 1
                    b_Hits50 += 1
                    b_Hits100 += 1
                elif rank < 50:
                    b_Hits50 += 1
                    b_Hits100 += 1
                elif rank < 100:
                    b_Hits100 += 1
                else:
                    continue

        return b_Hits1, b_Hits10, b_Hits50, b_Hits100
