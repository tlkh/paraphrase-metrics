# paraphrase-metrics

Metrics for better understanding of Paraphrases

## About

`pip install paraphrase-metrics`

A python package containing implementations of word position deviation (WPD) and lexical deviation (LD) proposed in "Towards Better Characterization of Paraphrases" (ACL 2022). 

* simple usage demo: [demo.ipynb](demo.ipynb)
* Streamlit demo: [tlkh/textdiff](https://huggingface.co/spaces/tlkh/textdiff)

## Paper

### Abstract

To effectively characterize the nature of paraphrase pairs without expert human annotation, we proposes two new metrics: word position deviation (WPD) and lexical deviation (LD). WPD measures the degree of structural alteration, while LD measures the difference in vocabulary used. We apply these metrics to better understand the commonly-used MRPC dataset and study how it differs from PAWS, another paraphrase identification dataset. We also perform a detailed study on MRPC and propose improvements to the dataset, showing that it improves generalizability of models trained on the dataset. Lastly, we apply our metrics to filter the output of a paraphrase generation model and show how it can be used to generate specific forms of paraphrases for data augmentation or robustness testing of NLP models. 

The paper can be found [here](https://aclanthology.org/2022.acl-long.588/).

### Talk

YouTube link: https://www.youtube.com/watch?v=rJlz96ICDEg&ab_channel=TimothyLiu

### Code

The full code associated with the ACL 2022 paper "Towards Better Characterization of Paraphrases" can be found in the `./paper/` folder. 

### Demo

Streamlit app showing MRPC/MRPC-R1 dataset filtering can be found at: [tlkh/paraphrase-metrics-mrpc](https://huggingface.co/spaces/tlkh/paraphrase-metrics-mrpc)

### Citing

```
@inproceedings{liu-soh-2022-towards,
    title = "Towards Better Characterization of Paraphrases",
    author = "Liu, Timothy  and
      Soh, De Wen",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.588",
    pages = "8592--8601",
}
```
