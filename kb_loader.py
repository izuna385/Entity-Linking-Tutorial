import faiss
import numpy as np

class KBIndexerWithFaiss:
    def __init__(self, config, entity_idx2emb, kbemb_dim=768):
        self.config = config
        self.kbemb_dim = kbemb_dim
        self.article_num = len(entity_idx2emb)
        self.entity_idx2emb = entity_idx2emb
        self.search_method_for_faiss = self.config.search_method_for_faiss
        self._indexed_faiss_loader()
        self.KBmatrix, self.kb_idx2entity_idx = self._KBmatrixloader()
        self._indexed_faiss_KBemb_adder(KBmatrix=self.KBmatrix)

    def _KBmatrixloader(self):
        KBemb = np.random.randn(self.article_num, self.kbemb_dim).astype('float32')
        kb_idx2entity_idx = {}
        for idx, (entity_idx, emb) in enumerate(self.entity_idx2emb.items()):
            KBemb[idx] = emb
            kb_idx2entity_idx.update({idx: entity_idx})

        return KBemb, kb_idx2entity_idx

    def _indexed_faiss_loader(self):
        if self.search_method_for_faiss == 'indexflatl2':  # L2
            self.indexed_faiss = faiss.IndexFlatL2(self.kbemb_dim)
        elif self.search_method_for_faiss == 'indexflatip':  #
            self.indexed_faiss = faiss.IndexFlatIP(self.kbemb_dim)
        elif self.search_method_for_faiss == 'cossim':  # innerdot * Beforehand-Normalization must be done.
            self.indexed_faiss = faiss.IndexFlatIP(self.kbemb_dim)

    def _indexed_faiss_KBemb_adder(self, KBmatrix):
        if self.search_method_for_faiss == 'cossim':
            KBemb_normalized_for_cossimonly = np.random.randn(self.article_num, self.kbemb_dim).astype('float32')
            for idx, emb in enumerate(KBmatrix):
                if np.linalg.norm(emb, ord=2, axis=0) != 0:
                    KBemb_normalized_for_cossimonly[idx] = emb / np.linalg.norm(emb, ord=2, axis=0)
            self.indexed_faiss.add(KBemb_normalized_for_cossimonly)
        else:
            self.indexed_faiss.add(KBmatrix)

    def _indexed_faiss_returner(self):
        return self.indexed_faiss