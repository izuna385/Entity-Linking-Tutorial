# Entity-Linking-Tutorial
* Tutorial for Bi-encoder based Entity Linking
![biencoder](./docs/candidate_biencoder.png)

# Environment Setup
* First, create base environment with conda.
```
$ conda create -n allennlp python=3.7
$ conda activate allennlp
$ pip install -r requirements.txt
```
* Second, download preprocessed files from [here](https://drive.google.com/drive/folders/1P-iXskc-hbqXateWh3wRknni_knqsagN?usp=sharing), then unzip.

* Third, run `python3 preprocess_mesh.py`

# Experiment and Evaluation
`python3 main.py`

# Docs for English
WIP

# Docs for Japanese
https://qiita.com/izuna385/items/9d658620b9b96b0b4ec9
https://qiita.com/izuna385/items/c2918874fbb564acf1e0
https://qiita.com/izuna385/items/367b7b365a2791ee4f8e

# LICENSE
MIT