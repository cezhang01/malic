import numpy as np
import scipy as sp
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize, Bounds, SR1
from scipy.sparse import csc_matrix
from scipy.special import softmax
from scipy.spatial.distance import cosine
import time
from evaluation import *


class Model():

    def __init__(self, args, data):

        self.parse_args(args, data)
        self.show_config()
        self.init_parameters()

    def parse_args(self, args, data):

        self.data = data
        self.dataset_name = args.dataset_name
        self.num_documents = self.data.num_documents
        self.num_words = self.data.num_words
        self.num_topics = args.num_topics
        self.num_unique_types = self.data.num_unique_types
        if args.num_aspects == None:
            self.num_unique_aspects = self.data.num_unique_aspects
        else:
            self.num_unique_aspects = args.num_aspects
        self.unique_types = self.data.unique_types
        self.unique_aspects = self.data.unique_aspects
        self.kendalls_tau_among_aspects_avg = self.data.kendalls_tau_among_aspects_avg
        self.pcc_among_aspects_avg = self.data.pcc_among_aspects_avg

        self.em_iter = args.em_iter
        self.qn_iter = args.qn_iter
        self.alpha_1 = args.alpha
        self.alpha_2 = args.alpha
        self.sigma_1 = args.sigma
        self.sigma_2 = args.sigma
        self.sigma_3 = args.sigma
        self.r = args.regularizer

        self.supervised_unsupervised = args.supervised_unsupervised
        self.training_ratio_documents = args.training_ratio_documents
        self.num_training_documents = len(self.data.training_documents)
        self.random_seed = args.random_seed

    def show_config(self):

        print('******************************************************')
        print('numpy version:', np.__version__)
        print('scipy version:', sp.__version__)

        print('dataset name:', self.dataset_name)
        print('#total documents:', self.num_documents)
        print('#words:', self.num_words)
        print('#topics:', self.num_topics)
        print('#unique partial ranking types:', self.num_unique_types)
        print('unique partial ranking types:', self.unique_types)
        print('#unique aspects:', self.num_unique_aspects)
        print('unique aspects:', set(self.unique_aspects))
        print('Normalized Kendall\'s tau among aspects:', self.kendalls_tau_among_aspects_avg)
        print('PCC among aspects:', self.pcc_among_aspects_avg)

        print('#EM iterations:', self.em_iter)
        print('#Quasi-Newton iterations:', self.qn_iter)
        print('alpha:', self.alpha_1)
        print('sigma:', self.sigma_1)
        print('regularizer:', self.r)

        print('supervised or unsupervised:', self.supervised_unsupervised)
        print('training partial rankings ratio: %.2f' % (len(self.data.training_partial_rankings) / len(self.data.partial_rankings)))
        print('#training partial rankings:', len(self.data.training_partial_rankings))
        print('training documents ratio:', self.training_ratio_documents)
        print('#training documents:', self.num_training_documents)
        print('******************************************************')

    def init_parameters(self):

        # self.aspect_dist = np.random.dirichlet(alpha=np.full(self.num_unique_aspects, self.alpha_1))  # pi
        # self.topic_word_dist = np.random.dirichlet(alpha=np.full(self.num_words, self.alpha_2), size=self.num_topics)  # theta
        # self.doc_topic_dist_unnormalized = np.random.multivariate_normal(mean=np.zeros(self.num_topics), cov=(1. / self.sigma_3) * np.eye(self.num_topics), size=self.num_training_documents)  # beta
        # self.doc_topic_dist = softmax(self.doc_topic_dist_unnormalized, axis=1)
        # self.global_w = np.random.multivariate_normal(mean=np.zeros(self.num_topics), cov=(1. / self.sigma_1) * np.eye(self.num_topics))
        # self.aspect_w = np.random.multivariate_normal(mean=self.global_w, cov=(1. / self.sigma_2) * np.eye(self.num_topics), size=self.num_unique_aspects)
        # self.balanced_para = np.random.uniform(low=0, high=1, size=self.num_unique_aspects)  # delta
        #self.balanced_para = np.ones(size=self.num_unique_aspects)  # delta

        self.aspect_dist = np.random.dirichlet(alpha=np.full(self.num_unique_aspects, 1))  # pi
        self.topic_word_dist = np.random.dirichlet(alpha=np.full(self.num_words, 1), size=self.num_topics)  # theta
        self.doc_topic_dist_unnormalized = np.random.multivariate_normal(mean=np.zeros(self.num_topics), cov=np.eye(self.num_topics), size=self.num_training_documents)  # beta
        self.doc_topic_dist = softmax(self.doc_topic_dist_unnormalized, axis=1)
        self.global_w = np.random.multivariate_normal(mean=np.zeros(self.num_topics), cov=np.eye(self.num_topics))
        self.aspect_w = np.random.multivariate_normal(mean=self.global_w, cov=np.eye(self.num_topics), size=self.num_unique_aspects)
        self.balanced_para = np.ones(self.num_unique_aspects)  # delta

    def e_step(self):

        # evaluate posterior distribution of topics
        self.posterior_prob_topic = []
        for doc_id, doc in enumerate(self.data.training_documents):
            posterior_prob_topic_one_doc = []
            for word_id in doc:
                numerators = np.multiply(self.doc_topic_dist[doc_id], self.topic_word_dist[:, word_id])
                posterior_prob_topic_one_word = numerators / np.sum(numerators)
                posterior_prob_topic_one_doc.append(posterior_prob_topic_one_word)
            posterior_prob_topic_one_doc = np.array(posterior_prob_topic_one_doc)
            self.posterior_prob_topic.append(posterior_prob_topic_one_doc)  # every element in this list is a N_i * num_topics array, where N_i is the number of words in document i

        # evaluate posterior distribution of each aspect that aspect or global w generates the partial ranking
        self.merit_values = self.evaluate_global_and_aspect_merit_values(self.global_w, self.aspect_w, self.doc_topic_dist)
        self.pl_probs = self.evaluate_pl_probs(self.merit_values, self.data.training_partial_rankings, self.data.training_types)
        balanced_para_repeat = np.tile(np.expand_dims(self.balanced_para, axis=0), [len(self.data.training_partial_rankings), 1])
        global_pl_prob_repeat = np.tile(np.expand_dims(self.pl_probs[:, 0], axis=1), [1, self.num_unique_aspects])
        numerators_c_is_0 = np.multiply(1 - balanced_para_repeat, global_pl_prob_repeat)
        numerators_c_is_1 = np.multiply(balanced_para_repeat, self.pl_probs[:, 1:])
        denominators_c = numerators_c_is_0 + numerators_c_is_1
        self.posterior_prob_w = np.array([np.divide(numerators_c_is_0, denominators_c), np.divide(numerators_c_is_1, denominators_c)])  # [2, num_training_partial_ranking, num_unique_aspects] where [0, :, :] is global and [1, :, :] is aspect-specific

        # evaluate posterior distribution of aspects
        if self.supervised_unsupervised == 'u':
            aspect_dist_repeat = np.tile(np.expand_dims(self.aspect_dist, axis=0), [len(self.data.training_partial_rankings), 1])
            numerators = np.multiply(aspect_dist_repeat, denominators_c)
            denominators_repeat = np.tile(np.expand_dims(np.sum(numerators, axis=1), axis=1), [1, self.num_unique_aspects])
            self.posterior_prob_aspect = np.divide(numerators, denominators_repeat)
        elif self.supervised_unsupervised == 's':
            one_hot = np.zeros([len(self.data.training_partial_rankings), self.num_unique_aspects])
            for partial_ranking_id in range(len(self.data.training_partial_rankings)):
                one_hot[partial_ranking_id, self.unique_aspects.index(self.data.training_aspects[partial_ranking_id])] = 1
            self.posterior_prob_aspect = np.copy(one_hot)

        return self.posterior_prob_topic, self.posterior_prob_aspect, self.posterior_prob_w

    def evaluate_global_and_aspect_merit_values(self, global_w, aspect_w, doc_topic_dist):

        merit_values = []
        for doc_id in range(len(doc_topic_dist)):
            merit_value_one_doc = np.exp(np.matmul(global_w, doc_topic_dist[doc_id]))
            merit_value_one_doc = np.concatenate([[merit_value_one_doc], np.exp(np.matmul(aspect_w, doc_topic_dist[doc_id]))])
            merit_values.append(merit_value_one_doc)
        merit_values = np.array(merit_values)  # [num_doc, num_unique_aspects + 1], each doc has num_unique_aspects + 1 merit values, the first is global, others are aspect-specific

        return merit_values

    def evaluate_pl_probs(self, merit_values, partial_rankings, types):

        pl_probs = []
        for partial_ranking_id, partial_ranking in enumerate(partial_rankings):
            if 'way' in types[partial_ranking_id]:
                numerators = merit_values[partial_ranking]
                denominators = np.array([np.sum(numerators[idx:, :], axis=0) for idx in range(len(partial_ranking))])
                pl_prob = np.prod(np.divide(numerators, denominators), axis=0)
                pl_probs.append(pl_prob)
            elif 'choice' in types[partial_ranking_id]:
                numerators = merit_values[partial_ranking][0, :]
                denominators = np.sum(merit_values[partial_ranking], axis=0)
                pl_prob = np.divide(numerators, denominators)
                pl_probs.append(pl_prob)
            elif 'top' in types[partial_ranking_id]:
                num_ranked_docs = int(types[partial_ranking_id][4:])  # L, number of ranked documents
                partial_merit_values = merit_values[partial_ranking]
                numerators = merit_values[partial_ranking[:num_ranked_docs]]
                denominators = np.array([np.sum(partial_merit_values[idx:, :], axis=0) for idx in range(num_ranked_docs)])
                pl_prob = np.prod(np.divide(numerators, denominators), axis=0)
                pl_probs.append(pl_prob)
        pl_probs = np.array(pl_probs)  # [num_training_partial_rankings, num_unique_aspects + 1], each partial ranking has num_unique_aspects + 1 pl probs, the first is global, others are aspect-specific

        return pl_probs

    def m_step(self):

        self.update_aspect_dist()
        self.update_topic_word_dist()

        # update doc_topic_dist_unnormalized and global and aspect w
        x0 = np.concatenate([self.doc_topic_dist_unnormalized.flatten(), self.global_w, self.aspect_w.flatten()])
        result = minimize(self.evaluate_expected_complete_data_log_likelihood, x0=x0, method='L-BFGS-B', options={'maxiter': self.qn_iter, 'disp': False}, jac=True)['x']
        self.doc_topic_dist_unnormalized = np.reshape(result[:(self.num_training_documents * self.num_topics)], [self.num_training_documents, self.num_topics])
        self.doc_topic_dist = softmax(self.doc_topic_dist_unnormalized, axis=1)
        self.global_w = result[(self.num_training_documents * self.num_topics):(self.num_training_documents * self.num_topics + self.num_topics)]
        self.aspect_w = np.reshape(result[(self.num_training_documents * self.num_topics + self.num_topics):], [self.num_unique_aspects, self.num_topics])

        # evaluate log-likelihood to check convergence
        self.evaluate_log_likelihood()

        return

    def update_aspect_dist(self):  # update pi

        numerators = np.sum(self.posterior_prob_aspect, axis=0) + self.alpha_1
        self.aspect_dist = np.divide(numerators, np.sum(numerators))

        return self.aspect_dist

    def update_topic_word_dist(self):  # update theta

        numerators = []
        for word_id in range(self.num_words):
            numerators_one_word = np.zeros(self.num_topics)
            for doc_id in range(self.num_training_documents):
                if word_id in self.data.training_documents[doc_id]:
                    selected_words_mask = self.data.training_documents[doc_id] == word_id
                    if np.sum(selected_words_mask * 1) > 0:
                        numerators_one_word += np.sum(self.posterior_prob_topic[doc_id][selected_words_mask], axis=0)
            numerators.append(self.r * numerators_one_word + self.alpha_2)
        numerators = np.array(numerators)
        denominators_repeat = np.tile(np.expand_dims(np.sum(numerators, axis=0), axis=0), [self.num_words, 1])
        self.topic_word_dist = np.transpose(np.divide(numerators, denominators_repeat))

        # numerators_one_word = [[np.sum(self.posterior_prob_topic[doc_id][self.data.training_documents[doc_id] == word_id], axis=0) for doc_id in range(self.num_training_documents) if word_id in self.data.training_documents[doc_id]] for word_id in range(self.num_words)]
        # numerators = np.array([np.sum(numerators_one_word[word_id], axis=0) for word_id in range(self.num_words)]) + self.alpha_2
        # self.topic_word_dist = np.transpose(np.divide(numerators, np.sum(numerators, axis=0)))

        return self.topic_word_dist

    def update_balanced_para(self):  # update delta

        numerators = np.sum(np.multiply(self.posterior_prob_aspect, self.posterior_prob_w[1, :, :]), axis=0)
        denominators = np.sum(self.posterior_prob_aspect, axis=0)
        self.balanced_para = np.divide(numerators, denominators)

        return self.balanced_para

    def evaluate_expected_complete_data_log_likelihood(self, x0):  # evaluate Q

        doc_topic_dist_unnormalized = np.reshape(x0[:(self.num_training_documents * self.num_topics)], [self.num_training_documents, self.num_topics])
        global_w = np.copy(x0[(self.num_training_documents * self.num_topics):(self.num_training_documents * self.num_topics + self.num_topics)])
        aspect_w = np.reshape(x0[(self.num_training_documents * self.num_topics + self.num_topics):], [self.num_unique_aspects, self.num_topics])

        self.Q = 0
        # evaluate expected complete-data log-likelihood
        doc_topic_dist = softmax(doc_topic_dist_unnormalized, axis=1)
        for doc_id, doc in enumerate(self.data.training_documents):
            numerators = np.multiply(np.tile(np.expand_dims(doc_topic_dist[doc_id], axis=0), [len(doc), 1]), np.transpose(self.topic_word_dist[:, doc]))
            self.Q += self.r * np.sum(np.multiply(self.posterior_prob_topic[doc_id], np.log(numerators + 1e-20)))

        merit_values = self.evaluate_global_and_aspect_merit_values(global_w, aspect_w, doc_topic_dist)
        pl_probs = self.evaluate_pl_probs(merit_values, self.data.training_partial_rankings, self.data.training_types)
        global_pl_prob_repeat = np.tile(np.expand_dims(pl_probs[:, 0], axis=1), [1, self.num_unique_aspects])
        c_is_0 = np.multiply(self.posterior_prob_w[0, :, :], np.log(global_pl_prob_repeat + 1e-20))
        c_is_1 = np.multiply(self.posterior_prob_w[1, :, :], np.log(pl_probs[:, 1:] + 1e-20))
        self.Q += np.sum(np.multiply(self.posterior_prob_aspect, c_is_0 + c_is_1))

        # evaluate log prior
        self.Q += np.sum(- 0.5 * self.sigma_3 * np.sum(np.square(doc_topic_dist_unnormalized), axis=1))
        self.Q += - 0.5 * self.sigma_1 * np.sum(np.square(global_w))
        self.Q += np.sum(- 0.5 * self.sigma_2 * np.sum(np.square(aspect_w - global_w), axis=1))

        # evaluate gradients
        gradients = []
        gradients.extend(self.evaluate_gradients_wrt_doc_topic_dist_unnormalized(doc_topic_dist_unnormalized, doc_topic_dist, global_w, aspect_w, merit_values))
        gradients.extend(self.evaluate_gradients_wrt_global_and_aspect_w(global_w, aspect_w, doc_topic_dist, merit_values))
        gradients = np.array(gradients)

        return - self.Q, - gradients

    def evaluate_gradients_wrt_doc_topic_dist_unnormalized(self, doc_topic_dist_unnormalized, doc_topic_dist, global_w, aspect_w, merit_values):

        gradients = []
        for doc_id, doc in enumerate(self.data.training_documents):
            gradients_one_doc = self.r * np.sum(self.posterior_prob_topic[doc_id] - doc_topic_dist[doc_id], axis=0)
            for partial_ranking_id, partial_ranking in enumerate(self.data.training_partial_rankings):
                if doc_id in partial_ranking:
                    gradients_scalar = 0
                    if 'way' in self.data.training_types[partial_ranking_id]:
                        rank_position = partial_ranking.index(doc_id)  # n, suppose doc_id in ranked at the n-th position
                        partial_merit_values = merit_values[partial_ranking]
                        denominators = np.array([np.sum(partial_merit_values[idx:, :], axis=0) for idx in range(rank_position + 1)])
                        gradients_scalar = 1 - np.sum(np.divide(partial_merit_values[rank_position], denominators), axis=0)
                    elif 'choice' in self.data.training_types[partial_ranking_id]:
                        chosen_or_not = (partial_ranking[0] == doc_id) * 1
                        doc_merit_value = merit_values[doc_id]
                        partial_merit_values = merit_values[partial_ranking]
                        denominators = np.sum(partial_merit_values, axis=0)
                        gradients_scalar = chosen_or_not - np.divide(doc_merit_value, denominators)
                    elif 'top' in self.data.training_types[partial_ranking_id]:
                        num_ranked_docs = int(self.data.training_types[partial_ranking_id][4:])  # L, number of ranked documents
                        rank_position = partial_ranking.index(doc_id)
                        if rank_position + 1 <= num_ranked_docs:
                            partial_merit_values = merit_values[partial_ranking]
                            denominators = np.array([np.sum(partial_merit_values[idx:, :], axis=0) for idx in range(rank_position + 1)])
                            gradients_scalar = 1 - np.sum(np.divide(partial_merit_values[rank_position], denominators), axis=0)
                        else:
                            partial_merit_values = merit_values[partial_ranking]
                            denominators = np.array([np.sum(partial_merit_values[idx:, :], axis=0) for idx in range(num_ranked_docs)])
                            gradients_scalar = - np.sum(np.divide(merit_values[doc_id], denominators), axis=0)
                    doc_topic_dist_repeat = np.tile(np.expand_dims(doc_topic_dist[doc_id], axis=1), [1, self.num_unique_aspects + 1])
                    gradients_pl = np.multiply(np.multiply(np.transpose(np.concatenate([np.expand_dims(global_w, axis=0), aspect_w], axis=0)) - np.log(merit_values[doc_id]), doc_topic_dist_repeat), gradients_scalar)  # [num_topics, num_unique_aspects + 1]
                    global_gradients_pl_repeat = np.tile(np.expand_dims(gradients_pl[:, 0], axis=1), [1, self.num_unique_aspects])
                    c_is_0 = np.multiply(self.posterior_prob_w[0, partial_ranking_id, :], global_gradients_pl_repeat)
                    c_is_1 = np.multiply(self.posterior_prob_w[1, partial_ranking_id, :], gradients_pl[:, 1:])
                    gradients_one_doc += np.sum(np.multiply(self.posterior_prob_aspect[partial_ranking_id], c_is_0 + c_is_1), axis=1)
            # add derivatives of log prior
            gradients_one_doc -= self.sigma_3 * doc_topic_dist_unnormalized[doc_id]
            gradients.extend(gradients_one_doc)

        return gradients

    def evaluate_gradients_wrt_global_and_aspect_w(self, global_w, aspect_w, doc_topic_dist, merit_values):

        gradients, gradients_global_w, gradients_aspect_w = [], 0, 0
        for partial_ranking_id, partial_ranking in enumerate(self.data.training_partial_rankings):
            gradients_pl = 0
            if 'way' in self.data.training_types[partial_ranking_id]:
                partial_merit_values = merit_values[partial_ranking]
                denominators = np.array([np.sum(partial_merit_values[idx:, :], axis=0) for idx in range(len(partial_ranking))])
                partial_doc_topic_dist = doc_topic_dist[partial_ranking]
                numerators = np.array([np.multiply(np.transpose(partial_doc_topic_dist), partial_merit_values[:, idx]) for idx in range(self.num_unique_aspects + 1)])  # [num_uniques_aspects + 1, num_topics, num_partial_ranking_documents]
                numerators = np.array([np.transpose([np.sum(numerators[aspect_idx, :, idx:], axis=1) for idx in range(len(partial_ranking))]) for aspect_idx in range(self.num_unique_aspects + 1)])
                gradients_pl = np.sum(np.array([np.transpose(partial_doc_topic_dist) - np.divide(numerators[idx], denominators[:, idx]) for idx in range(self.num_unique_aspects + 1)]), axis=2)  # [num_unique_aspects + 1, num_topics]
            elif 'choice' in self.data.training_types[partial_ranking_id]:
                partial_merit_values = merit_values[partial_ranking]
                denominators = np.sum(partial_merit_values, axis=0)
                partial_doc_topic_dist = doc_topic_dist[partial_ranking]
                numerators = np.matmul(np.transpose(partial_doc_topic_dist), partial_merit_values)  # [num_topics, num_unique_aspects + 1]
                gradients_pl = np.array([partial_doc_topic_dist[0] - np.divide(numerators[:, idx], denominators[idx]) for idx in range(self.num_unique_aspects + 1)])  # [num_unique_aspects + 1, num_topics]
            elif 'top' in self.data.training_types[partial_ranking_id]:
                num_ranked_docs = int(self.data.training_types[partial_ranking_id][4:])  # L, number of ranked documents
                partial_merit_values = merit_values[partial_ranking]
                denominators = np.array([np.sum(partial_merit_values[idx:, :], axis=0) for idx in range(num_ranked_docs)])
                partial_doc_topic_dist = doc_topic_dist[partial_ranking]
                numerators = np.array([np.multiply(np.transpose(partial_doc_topic_dist), partial_merit_values[:, idx]) for idx in range(self.num_unique_aspects + 1)])
                numerators = np.array([np.transpose([np.sum(numerators[aspect_idx, :, idx:], axis=1) for idx in range(num_ranked_docs)]) for aspect_idx in range(self.num_unique_aspects + 1)])
                gradients_pl = np.sum(np.array([np.transpose(partial_doc_topic_dist[:num_ranked_docs]) - np.divide(numerators[idx], denominators[:, idx]) for idx in range(self.num_unique_aspects + 1)]), axis=2)
            # evaluate gradients wrt global_w
            gradients_global_w += np.matmul(self.posterior_prob_aspect[partial_ranking_id], self.posterior_prob_w[0, partial_ranking_id]) * gradients_pl[0]
            # evaluate gradients wrt aspect_w
            prod_of_responsibilities = np.multiply(self.posterior_prob_aspect[partial_ranking_id], self.posterior_prob_w[1, partial_ranking_id])
            gradients_aspect_w += np.multiply(prod_of_responsibilities, np.transpose(gradients_pl[1:])).flatten('F')
        # add log priors
        gradients_global_w -= self.sigma_1 * global_w
        gradients_aspect_w -= self.sigma_2 * (aspect_w - global_w).flatten()
        gradients.extend(gradients_global_w)
        gradients.extend(gradients_aspect_w)

        return gradients

    def evaluate_log_likelihood(self):

        self.ll = 0
        # evaluate document log-likelihood
        for doc_id, doc in enumerate(self.data.training_documents):
            self.ll += np.sum(np.log(np.matmul(np.transpose(self.topic_word_dist[:, doc]), self.doc_topic_dist[doc_id])))

        # evaluate partial ranking log-likelihood
        self.merit_values = self.evaluate_global_and_aspect_merit_values(self.global_w, self.aspect_w, self.doc_topic_dist)
        pl_probs = self.evaluate_pl_probs(self.merit_values, self.data.training_partial_rankings, self.data.training_types)
        global_pl_prob_repeat = np.tile(np.expand_dims(pl_probs[:, 0], axis=1), [1, self.num_unique_aspects])
        self.ll += np.sum(np.log(np.sum(np.multiply(self.balanced_para * pl_probs[:, 1:] + (1 - self.balanced_para) * global_pl_prob_repeat, self.aspect_dist), axis=0)))

        # add log priors
        self.ll += np.sum(self.alpha_1 * np.log(self.aspect_dist + 1e-20))
        self.ll += np.sum(self.alpha_2 * np.log(self.topic_word_dist + 1e-20))
        self.ll += np.sum(- 0.5 * self.sigma_3 * np.sum(np.square(self.doc_topic_dist_unnormalized), axis=1))
        self.ll += - 0.5 * self.sigma_1 * np.sum(np.square(self.global_w))
        self.ll += np.sum(- 0.5 * self.sigma_2 * np.sum(np.square(self.aspect_w - self.global_w), axis=1))

        return self.ll

    def test(self):

        if self.training_ratio_documents < 1:
            self.evaluate_topic_dist_of_test_documents()
            self.aspect_and_ranking_prediction()
            self.evaluate_perplexity()
        else:
            self.rank_aggregation_prediction()
        # self.evaluate_correlation_among_w()

    def aspect_and_ranking_prediction(self):

        # evaluate posterior probability of global_w and aspect_w for every aspect given test partial rankings
        merit_values = self.evaluate_global_and_aspect_merit_values(self.global_w, self.aspect_w, self.doc_topic_dist_total)
        pl_probs = self.evaluate_pl_probs(merit_values, self.data.test_partial_rankings, self.data.test_types)
        balanced_para_repeat = np.tile(np.expand_dims(self.balanced_para, axis=0), [len(self.data.test_partial_rankings), 1])
        global_pl_prob_repeat = np.tile(np.expand_dims(pl_probs[:, 0], axis=1), [1, self.num_unique_aspects])
        numerators_c_is_0 = np.multiply(1 - balanced_para_repeat, global_pl_prob_repeat)
        numerators_c_is_1 = np.multiply(balanced_para_repeat, pl_probs[:, 1:])
        denominators_c = numerators_c_is_0 + numerators_c_is_1

        # evaluate posterior probability of each aspect given test partial rankings
        aspect_dist_repeat = np.tile(np.expand_dims(self.aspect_dist, axis=0), [len(self.data.test_partial_rankings), 1])
        numerators = np.multiply(aspect_dist_repeat, denominators_c)
        denominators_repeat = np.tile(np.expand_dims(np.sum(numerators, axis=1), axis=1), [1, self.num_unique_aspects])
        predicted_aspect_probs = np.divide(numerators, denominators_repeat)
        predicted_aspects_numeric = np.argmax(predicted_aspect_probs, axis=1)

        # evaluate partial merit values of test partial rankings
        partial_merit_values = [merit_values[partial_ranking] for partial_ranking in self.data.test_partial_rankings]  # [num_test_partial_rankings, num_docs_in_each_partial_ranking, num_unique_aspects + 1]
        if self.supervised_unsupervised == 's':
            predicted_merit_values = [partial_merit_values[partial_ranking_idx][:, (self.data.test_aspects_numeric[partial_ranking_idx] + 1)]
                                      for partial_ranking_idx, partial_ranking in enumerate(self.data.test_partial_rankings)]
            aspect_prediction(self.data.test_aspects_numeric, predicted_aspects_numeric, self.supervised_unsupervised)
        elif self.supervised_unsupervised == 'u':
            predicted_merit_values = [partial_merit_values[partial_ranking_idx][:, (predicted_aspects_numeric[partial_ranking_idx] + 1)]
                                      for partial_ranking_idx, partial_ranking in enumerate(self.data.test_partial_rankings)]

        # making predictions
        ranking_prediction_kendalls_tau(predicted_merit_values, self.data.test_types)

    def rank_aggregation_prediction(self):

        merit_values = self.evaluate_global_and_aspect_merit_values(self.global_w, self.aspect_w, self.doc_topic_dist)
        rank_aggregation_prediction(merit_values, self.dataset_name, self.unique_aspects)

    def evaluate_topic_dist_of_test_documents(self):

        self.doc_topic_dist_test = []
        for doc_id, doc in enumerate(self.data.test_documents):
            numerators = np.mean(np.multiply(self.doc_topic_dist, np.prod(self.topic_word_dist[:, doc], axis=1) + 1e-20), axis=0)
            self.doc_topic_dist_test.append(numerators / np.sum(numerators))
        self.doc_topic_dist_test = np.array(self.doc_topic_dist_test)
        self.doc_topic_dist_total = np.concatenate([self.doc_topic_dist, self.doc_topic_dist_test], axis=0)

    def evaluate_perplexity(self):

        pp, num_words_in_docs = 0, 0
        for doc_id, doc in enumerate(self.data.test_documents):
            pp += np.sum(np.log(np.matmul(np.transpose(self.topic_word_dist[:, doc]), self.doc_topic_dist_test[doc_id]) + 1e-20))
            num_words_in_docs += len(doc)
        print('Perplexity = %.4f' % np.exp(- pp / num_words_in_docs))
        print('Per-word log-likelihood = %.4f' % (- pp / num_words_in_docs))

    def output_top_words_of_all_topics(self):

        num_top_words = 10
        top_words_idx = np.flip(np.argsort(self.topic_word_dist, axis=1)[:, -num_top_words:], axis=1)
        self.top_words = self.data.voc[top_words_idx]
        print('Top words of each topic****************************************************************')
        print(self.top_words)

    def output_top_words_of_aspect_related_topics(self):

        num_topics_each_aspect = 5
        topic_id = np.flip(np.argsort(self.aspect_w, axis=1)[:, -num_topics_each_aspect:], axis=1)
        for aspect_id, aspect in enumerate(self.unique_aspects):
            print('Aspect id:', aspect_id, 'Aspect:', aspect, '****************************************************************')
            for i in range(num_topics_each_aspect):
                print('Topic id:', topic_id[aspect_id, i], 'Keywords:', self.top_words[topic_id[aspect_id, i]])

    def evaluate_correlation_among_w(self):

        cosine_similarity = []
        for aspect_idx1 in range(len(self.unique_aspects) - 1):
            for aspect_idx2 in range(aspect_idx1 + 1, len(self.unique_aspects)):
                cosine_similarity.append(1 - cosine(self.aspect_w[aspect_idx1], self.aspect_w[aspect_idx2]))
        pcc = pearsonr(self.data.kendalls_tau_among_aspects, cosine_similarity)[0]
        print('Correlation among aspects: PCC = %.4f' % pcc)

    def save_results(self):

        np.savetxt('./results/' + self.dataset_name + '/doc_topic_dist_' + str(self.num_topics) + '.txt', self.doc_topic_dist, fmt='%f', delimiter=' ')
        np.savetxt('./results/' + self.dataset_name + '/topic_word_dist_' + str(self.num_topics) + '.txt', self.topic_word_dist, fmt='%f', delimiter=' ')

    def train(self):

        t = time.time()
        for epoch_idx in range(1, self.em_iter + 1):
            self.e_step()
            self.m_step()
            print('******************************************************')
            print('Time: %ds' % (time.time() - t), '\tEpoch: %d/%d' % (epoch_idx, self.em_iter), '\tLL: %f' % self.ll, '\tQ: %f' % (- self.Q))
            self.test()

        self.output_top_words_of_all_topics()
        self.output_top_words_of_aspect_related_topics()
        self.save_results()
        self.show_config()