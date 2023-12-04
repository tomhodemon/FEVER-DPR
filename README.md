# FEVER-DPR

**Initially a coursework for Deep Learning for Natural Language Processing (AI605)**

This repository contains: 
* Source code used to prepare and publish the simplified version of the FEVER dataset to HF Hub.
* Source code of a lightweight re-implementation of a [Dense Passage Retriever (DPR)](https://arxiv.org/pdf/2004.04906.pdf).
* Source code used to train the DPR.
* A set of results from training experiments.

## Process FEVER data
Once processed, the data follow the retriever input data format specified in the [official repository](https://github.com/facebookresearch/DPR) without the *answers* and *negative_ctxs* fields, which are not used in our context.

## Training experiments and evaluation results
The repository contains TensorBoard logs produced during training experiments as well as evaluation results. Recall@k is used as metric to evaluate and compare models.
Both can be visualize in details using:
```
tensorboard --logdir runs
```

