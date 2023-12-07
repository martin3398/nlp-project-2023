# NLP Project 2023

This repository contains our Natural Language Processing Term Project at Seoul National University

## Used Dataset

For this project, we use the Kaggle Fake News Dataset: [https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification).

## Finetuned Bert Classifier

We then finetune a Bert Classifier on the dataset.

## Style Transfer

We use the following library to perform a casual to formal style transfer on the fake news in the test set:

* [https://huggingface.co/rajistics/informal_formal_style_transfer](https://huggingface.co/rajistics/informal_formal_style_transfer)
* [https://github.com/PrithivirajDamodaran/Styleformer#active-to-passive--available-now---](https://github.com/PrithivirajDamodaran/Styleformer#active-to-passive--available-now---)

We then report score for on the test set without and with style transfer.

The hypothesis is that the classification of fake news gets worse.

## TODO

* [ ] Ensure reproducibility by using the same seed everywhere (there seems to be no seed somewhere - maybe torch)
