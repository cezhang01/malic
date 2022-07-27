# MALIC
Source code and datasets of CIKM-21 paper, Topic Modeling for Multi-Aspect Listwise Comparisons, by [Delvin Ce Zhang](http://www.delvincezhang.com) and [Hady W. Lauw](http://www.hadylauw.com).

MALIC is a topic model that can incorporate document comparisons as meta-data to improve topic modeling quality and produce a list of document comparison outcomes.

## Implementation Environment
- numpy == 1.17.4
- scipy == 1.3.1

## Run
`python main.py`

### Parameter Setting
- -em: number of EM iterations, default = 20
- -qn: number of iterations of Quasi-Newton method, default = 20
- -dn: dataset name, default = country
- -nt: number of topics, default = 30
- -su: s = supervised setting, u = unsupervised setting, default = s
- -tr2: ratio of training documents, 1 = rank aggregation, 0.8 = partial ranking prediction, aspect assignment, and perplexity, default = 1
- -na: number of aspects, None = use the correct number of aspects in the dataset, other positive integers (1 or 4 or 8 or 16) can be set, default = None
- -p: number of documents added to existing partial rankings, 0 (used in the main paper) = use {3, 4, 5}-way, top-{2, 3, 4}, and choice-{5, 10, 15}, other positive integers (2 or 4 or 6 or 8) can be set, default = 0
- -e: number of partial rankings for each length and each aspect, default = 50, change to 5 or 25 or 100 or 200
- -a: alpha, Dirichlet prior, default = 0.01
- -s: sigma, Gaussian covariance prior, default = 0.01
- -r: lambda, regularizer, default = 0.01
- -rs: random seed, we randomly generate 5 different random seeds to run experiments independently, and report both mean and standard deviation in the main paper

## Output
Results will be output to `./results` file.
- `doc_topic_dist.txt` contains #documents row, each row is a topic distribution with #topics diminsions
- `topic_word_dist.txt` contains #topics row, each row is a distribution over #words words

## Reference
If you use our paper, including code and data, please cite

```
@inproceedings{malic,
  title={Topic Modeling for Multi-Aspect Listwise Comparisons},
  author={Zhang, Delvin Ce and Lauw, Hady W},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={2507--2516},
  year={2021}
}
```
