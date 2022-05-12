# Towards Better Characterization of Paraphrases

Code for the paper: Towards Better Characterization of Paraphrases

To effectively characterize the nature of paraphrase pairs without expert human annotation, we proposes two new metrics: word position deviation (WPD) and lexical deviation (LD). WPD measures the degree of structural alteration, while LD measures the difference in vocabulary used. We apply these metrics to better understand the commonly-used MRPC dataset and study how it differs from PAWS, another paraphrase identification dataset. We also perform a detailed study on MRPC and propose improvements to the dataset, showing that it improves generalizability of models trained on the dataset. Lastly, we apply our metrics to filter the output of a paraphrase generation model and show how it can be used to generate specific forms of paraphrases for data augmentation or robustness testing of NLP models. 

## Contents

* `classifier`: contains code used to train classification models on MRPC, MRPC-R1 and PAWS
* `datasets`: contains the code for dataset processing
* `generation`: contains the code used for training seq2seq language models on paraphrase corpus

## Setup

* Environment: NGC PyTorch container: `nvcr.io/nvidia/pytorch:21.08-py3`
* Setup script to install additional libraries inside container: `setup.sh`
* Datasets:
  * MRPC: instructions at `datasets/mrpc`
  * PAWS: instructions at `datasets/paws`

Recommended start-up steps:

```shell
docker pull nvcr.io/nvidia/pytorch:21.08-py3

docker run --rm \
  -p 8888:8888 \
  -v THIS_FOLDER:/workspace \
  nvcr.io/nvidia/pytorch:21.08-py3 jupyter lab
  
# in Jupyter Lab terminal
bash setup.sh
```
