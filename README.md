# NLP Project 2023

This repository contains our Natural Language Processing Term Project at Seoul National University

## Structure of the Project

We provide the following Jupyter Notebooks:

* `01-Clean_and_Split.ipynb`: Cleans the dataset and splits it into train and test set
* `02a-Styletransfer.ipynb`: Applies style transfer to the fake news articles of the test set
* `02b-Tokenize_Original.ipynb`: Tokenizes the original dataset (without styletransfer) for training and evaluation
* `03_Train.ipynb`: Finetunes a Bert Classifier on the original dataset
* `04_Evaluate.ipynb`: Evaluates the finetuned Bert Classifier on the original and style transferred test set

We furthermore provide a `00-Run_Everything.ipynb` Notebook that runs all the steps in the correct order.

## How to Run this Project

Please download the dataset and put `./dataset.csv`.

You can then execute the different notebooks in correct order.
Please note that this creates and exports some intermediate files into the `./data` folder.

## Project Description

This projects evaluates the impact of style transfer on the classification of fake news using Bert.

In short, we finetune Bert to classify news as fake or real.
We then use a style transfer model to transform the fake news into a more formal style.
The hypothesis tested is that the classification of fake news gets worse, which will be evaluated on the test set using the finetuned Bert Classifier.

### Used Dataset

For this project, we use the Kaggle Fake News Dataset: [https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification).

### Finetuned Bert Classifier

We then finetune a Bert ([https://huggingface.co/docs/transformers/model_doc/bert](https://huggingface.co/docs/transformers/model_doc/bert)) Classifier on the dataset.

### Style Transfer

We use the following library to perform a casual to formal style transfer on the fake news in the test set:

* [https://huggingface.co/rajistics/informal_formal_style_transfer](https://huggingface.co/rajistics/informal_formal_style_transfer)
* [https://github.com/PrithivirajDamodaran/Styleformer#active-to-passive--available-now---](https://github.com/PrithivirajDamodaran/Styleformer#active-to-passive--available-now---)

### Evaluation

We then report scores on the test set without and with style transfer.

The hypothesis is that the classification of fake news gets worse.

