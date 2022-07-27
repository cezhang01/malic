import numpy as np
from evaluation import *


class Data():

    def __init__(self, args):

        self.parse_args(args)
        self.load_data()

    def parse_args(self, args):

        self.dataset_name = args.dataset_name
        self.training_ratio_documents = args.training_ratio_documents
        self.partial_ranking_plus_length = args.partial_ranking_plus_length
        self.num_partial_rankings_each_length = args.num_partial_rankings_each_length

    def load_data(self):

        self.load_documents()
        self.load_partial_rankings()
        self.split_partial_rankings()
        if self.dataset_name == 'paper_review' or self.dataset_name == 'hotel_review' or self.dataset_name == 'employee_review' or self.dataset_name == 'hotel_review_large':
            self.load_overall_ranking()
        self.compute_kendalls_tau_among_aspects()

    def load_documents(self):

        total_documents, self.training_documents, self.test_documents = [], [], []

        with open('./data/' + self.dataset_name + '/documents.txt') as file:
            for row in file:
                row = row.split()
                document = np.array([int(word) for word in row])
                total_documents.append(document)
        self.num_documents = len(total_documents)

        for doc_id, doc in enumerate(total_documents):
            if doc_id <= int(self.num_documents * self.training_ratio_documents):
                self.training_documents.append(doc)
            else:
                self.test_documents.append(doc)

        self.voc = np.genfromtxt('./data/' + self.dataset_name + '/voc.txt', dtype=str, encoding='utf-8')
        self.num_words = len(self.voc)

    def load_partial_rankings(self):

        self.partial_rankings, self.types, self.aspects = [], [], []
        plus = '_plus_' + str(self.partial_ranking_plus_length) if self.partial_ranking_plus_length > 0 else ''
        each = '_' + str(self.num_partial_rankings_each_length) + '_each' if self.num_partial_rankings_each_length != 50 else ''
        with open('./data/' + self.dataset_name + '/partial_rankings' + plus + each + '.txt') as file:
            for row in file:
                row = row.split()
                self.types.append(row[0])
                self.aspects.append(row[1])
                self.partial_rankings.append([int(doc_id) for doc_id in row[2:]])
        self.types, self.aspects = np.array(self.types), np.array(self.aspects)
        self.unique_types, self.unique_aspects = np.unique(self.types), np.unique(self.aspects).tolist()
        self.num_unique_types, self.num_unique_aspects = len(self.unique_types), len(self.unique_aspects)

    def split_partial_rankings(self):

        self.training_partial_rankings, self.training_types, self.training_aspects = [], [], []
        self.test_partial_rankings, self.test_types, self.test_aspects = [], [], []
        for partial_ranking_idx, partial_ranking in enumerate(self.partial_rankings):
            if (np.array(partial_ranking) <= int(self.num_documents * self.training_ratio_documents)).__contains__(False):
                self.test_partial_rankings.append(partial_ranking)
                self.test_types.append(self.types[partial_ranking_idx])
                self.test_aspects.append(self.aspects[partial_ranking_idx])
            else:
                self.training_partial_rankings.append(partial_ranking)
                self.training_types.append(self.types[partial_ranking_idx])
                self.training_aspects.append(self.aspects[partial_ranking_idx])
        self.training_types, self.training_aspects, self.test_types, self.test_aspects = \
            np.array(self.training_types), np.array(self.training_aspects), np.array(self.test_types), np.array(self.test_aspects)
        self.test_aspects_numeric = np.array([self.unique_aspects.index(test_asepct) for test_asepct in self.test_aspects])

    def load_overall_ranking(self):

        self.overall_ranking = {}
        with open('./data/' + self.dataset_name + '/overall_ranking.txt') as file:
            for row in file:
                row = row.split()
                self.overall_ranking[int(row[0])] = int(row[1])

    def compute_kendalls_tau_among_aspects(self):

        self.kendalls_tau_among_aspects, self.pcc_among_aspects = evaluate_kendalls_tau_among_aspects(self.dataset_name, self.unique_aspects)
        self.kendalls_tau_among_aspects_avg, self.pcc_among_aspects_avg = np.mean(self.kendalls_tau_among_aspects), np.mean(self.pcc_among_aspects)