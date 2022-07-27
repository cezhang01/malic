import argparse
import numpy as np

from data_loader import Data
from model import Model


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('-em', '--em_iter', type=int, default=20)
    parser.add_argument('-qn', '--qn_iter', type=int, default=20)
    parser.add_argument('-dn', '--dataset_name', type=str, default='country')
    parser.add_argument('-nt', '--num_topics', type=int, default=30)
    parser.add_argument('-su', '--supervised_unsupervised', type=str, default='u')
    parser.add_argument('-tr2', '--training_ratio_documents', type=float, default=0.72)
    parser.add_argument('-na', '--num_aspects', type=int, default=None)
    parser.add_argument('-p', '--partial_ranking_plus_length', type=int, default=0)
    parser.add_argument('-e', '--num_partial_rankings_each_length', type=int, default=50)
    parser.add_argument('-a', '--alpha', type=float, default=0.01)
    parser.add_argument('-s', '--sigma', type=float, default=0.01)
    parser.add_argument('-r', '--regularizer', type=float, default=0.01)
    parser.add_argument('-rs', '--random_seed', type=int, default=519)

    return parser.parse_args()


def main(args):

    if args.random_seed:
        np.random.seed(args.random_seed)
    print('Preparing data...')
    data = Data(args)
    print('Initializing model...')
    model = Model(args, data)
    print('Start training...')
    model.train()


if __name__ == '__main__':
    main(parse_arguments())