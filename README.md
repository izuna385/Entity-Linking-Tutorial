# Entity-Linking-Tutorial
* Tutorial for Bi-encoder based Entity Linking
  - Surface-Candidate based
  
    ![biencoder](./docs/candidate_biencoder.png)
  
  - ANN-search based
  
    ![entire_biencoder](./docs/biencoder.png)

# Environment Setup
* First, create base environment with conda.
```
$ conda create -n allennlp python=3.7
$ conda activate allennlp
$ pip3 install -r requirements.txt
```
* Second, download preprocessed files from [here](https://drive.google.com/drive/folders/1P-iXskc-hbqXateWh3wRknni_knqsagN?usp=sharing), then unzip.

* Third, download [BC5CDR dataset](https://biocreative.bioinformatics.udel.edu/resources/corpora/biocreative-v-cdr-corpus/) to `./dataset/` and unzip.

* You have to place `CDR_DevelopmentSet.PubTator.txt`, `CDR_TestSet.PubTator.txt` and `CDR_TrainingSet.PubTator.txt` under `./dataset/`.

* Then, run `python3 BC5CDRpreprocess.py` and `python3 preprocess_mesh.py`.

# Model and Scoring
![scoring](./docs/scoring.png)
* Derived from [[Logeswaran et al., '19]](https://arxiv.org/abs/1906.07348)

# Experiment and Evaluation
```
$ rm -r serialization_dir # Remove pre-experiment result if you run `python3 main.py -debug` for debugging.
$ python3 main.py
```

# Parameters
We only here note critical parameters for training and evaluation. For further detail, see `parameters.py`.

| Parameter Name            | Description                                                                                                                                                                  | Default      |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|
| `batch_size_for_train`    | Batch size during learning. The more there are, the more the encoder will learn to choose the correct answer from more negative examples.                                    | `16`         |
| `lr`                      | Learning rate.                                                                                                                                                               | `1e-5`       |
| `max_candidates_num`      | Determine how many candidates are to be generated for each mention by using surface form.                                                                                    | `5`          |
| `search_method_for_faiss` | This specifies whether to use the cosine distance (`cossim`), inner product (`indexflatip`), or L2 distance (`indexflatl2`) when performing approximate neighborhood search. | `indexflatip`|


# Result
With default parameters.

* Surface-Candidate based
  
  | Generated Candidates Num | 5     | 10    | 20    |
  |--------------------------|-------|-------|-------|
  | dev_recall               | 76.80 | 79.91 | 80.92 |
  | dev_acc                  | 59.85 | 52.56 | 47.23 |
  | test_recall              | 74.35 | 77.14 | 78.25 |
  | test_acc                 | 58.51 | 51.38 | 45.69 |

* ANN-search Based 

  (Generated Candidates Num: 50 (Fixed))
  
  | Recall@X   | 1 (Acc.) | 5     | 10    | 50    |
  |------------|----------|-------|-------|-------|
  | dev_recall | 21.58    | 42.28 | 50.48 | 67.11 |
  | test_recall| 21.50    | 40.29 | 47.95 | 64.52 |


# Docs for English
* https://izuna385.medium.com/building-bi-encoder-based-entity-linking-system-with-transformer-6c111d86500

# Docs for Japanese
* [Part 1: History](https://qiita.com/izuna385/items/9d658620b9b96b0b4ec9)
* [Part 2: Preprocecssing](https://qiita.com/izuna385/items/c2918874fbb564acf1e0)
* [Part 3: Model and Evaluation](https://qiita.com/izuna385/items/367b7b365a2791ee4f8e)
* [Part 4: ANN-search with Faiss](https://qiita.com/izuna385/items/bce14031e8a443a0db44)
* [Sub Contents: Reproduction of experimental results using Colab-Pro](https://qiita.com/izuna385/items/bbac95594e20e6990189)

# LICENSE
MIT