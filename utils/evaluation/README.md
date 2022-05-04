# Evaluation metodology

To score the submissions, we are using the FITB metric.

**Fill In The Blank (FITB)** - This is the most commonly used metric for
offline evaluation of models that generate fashion outfits. The main goal of
this metric is to access the ability of the model to correctly predict the item
missing given an incomplete outfit. For each outfit from the
test dataset, we masked one item and use the rest to represent the incomplete_outfit.
The next step is to create a set of candidates containing various items. This set contains the masked item,
as well as, some other items. This is a proxy of the model’s capability on learning how to
combine items that share some coherence in terms of colours and patterns. The final score
of this metric is the ratio of correct predictions your model generates and the total number of test outfits.

To help you evaluate your model, this folder contains two scripts:

- fitb.py with a simple implementation of the FITB metric.
- simple_split_dataset.ipynb that is able to split the dataset between train and test, in order to you to evaluate your
  model.

To more information on how to use the dataset check the README.md in baseline folder.