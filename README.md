# verb-attributes
This repository contains data and code for the EMNLP 2017 paper [Zero-Shot Activity Recognition with Verb Attribute Induction](https://arxiv.org/abs/1707.09468). (For more information, see the paper). If you use our Verbs with Attributes corpus or if the paper significantly inspires you, we request that you cite our work:

### Bibtex

```
@inproceedings{emnlp17_zellers,
author = {Rowan Zellers and Yejin Choi},
title = {Zero-Shot Activity Recognition with Verb Attribute Induction},
url = {https://arxiv.org/abs/1707.09468},
booktitle = "Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
year = "2017",
}
```

## Obtaining the Verb Annotations

[Our annotations are available in this folder.](data/VerbsWithAttributes) See the readme there for more information.

To download the imSitu images, you'll need to follow the instructions from the install script [here.](https://github.com/my89/SituationCrf/tree/master)

## Dependencies

I originally wrote this code with PyTorch 1.12, but I've updated it to hopefully work for PyTorch 3.0.
See [requirements.txt](requirements.txt) for dependencies. For ease of use, I recommend installing everything in a virtualenv. Ping me if there's a dependency missing (I tried to prune dependencies not needed for this project in particular).

## Reproducing our results

Here's the rough outline to reproduce my results:

```
# Pretrain dictionary challenge
python models/def_to_atts_pretrain.py

# pretrain IMSITU
python models/imsitu_pretrain.py

# Train the ensembling text model
python models/def_to_atts_train.py

# Train imsitu
python models/imsitu_train.py
```

For evaluation, use the scripts ```def_to_atts_eval.py``` and ```imsitu_eval.py```

## Questions?

This documentation is a work in progress, so flag an issue or [contact me](http://rowanzellers.com/) if you have any questions.
