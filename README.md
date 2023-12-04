# NLP Project 2023

This repository contains the Natural Language Processing Project at Seoul National University

## Used Dataset

For this project, we use the Kaggle Fake News Dataset: [https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification).

## Finetuned Bert Classifier

We then finetune a Bert Classifier on the dataset.

## Style Transfer

We use the following library to perform a casual to formal style transfer on the fake news in the test set:

* [https://huggingface.co/rajistics/informal_formal_style_transfer](https://huggingface.co/rajistics/informal_formal_style_transfer)
* [https://github.com/PrithivirajDamodaran/Styleformer#active-to-passive--available-now---](https://github.com/PrithivirajDamodaran/Styleformer#active-to-passive--available-now---)

We then report score for on the test set without with style transfer.

The hypothesis is that the classification of fake news gets worse.

## TODO

Currently the Jupyter Notebook takes too long to run. We need to modularize the code and create scripts/notebooks that can be run independently.

* [ ] Create notebook that creates test and train set and performs style transfer on test set
* [ ] Create notebook that finetunes Bert Classifier on the dataset using the test/train set of the first notebook
* [ ] Create notebook that evaluates the Classifier on the test set with and without style transfer
* [ ] Ensure reproducibility by using the same seed everywhere (there seems to be no seed somewhere - maybe torch)
