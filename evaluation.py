import numpy as np
from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score, adjusted_rand_score
from scipy.stats import pearsonr, kendalltau


def aspect_prediction(test_aspects_numeric, predicted_aspects_numeric, supervised_unsupervised):

    acc = accuracy_score(test_aspects_numeric, predicted_aspects_numeric)
    f1 = f1_score(test_aspects_numeric, predicted_aspects_numeric, average='macro')
    print('Aspect Prediction: Acc = %.4f, F1 = %.4f' % (acc, f1))


def ranking_prediction_kendalls_tau(predicted_merit_values, test_types):

    kendalls_tau = 0
    for partial_ranking_id, type in enumerate(test_types):
        if 'top' in type:
            num_ranked_docs = int(type[4:])  # L, number of ranked documents
            num_partial_docs = len(predicted_merit_values[partial_ranking_id])
            concordant_pairs, discordant_pairs = 0, 0
            for ranked_doc_idx in range(num_ranked_docs):
                concordant_pairs += np.sum((predicted_merit_values[partial_ranking_id][ranked_doc_idx + 1:] < predicted_merit_values[partial_ranking_id][ranked_doc_idx]) * 1)
                discordant_pairs += np.sum((predicted_merit_values[partial_ranking_id][ranked_doc_idx + 1:] > predicted_merit_values[partial_ranking_id][ranked_doc_idx]) * 1)
            kendalls_tau += (((concordant_pairs - discordant_pairs) / (0.5 * num_ranked_docs * (2 * num_partial_docs - num_ranked_docs - 1))) + 1) / 2
        elif 'way' in type:
            num_partial_docs = len(predicted_merit_values[partial_ranking_id])
            concordant_pairs, discordant_pairs = 0, 0
            for ranked_doc_idx in range(num_partial_docs - 1):
                concordant_pairs += np.sum((predicted_merit_values[partial_ranking_id][ranked_doc_idx + 1:] < predicted_merit_values[partial_ranking_id][ranked_doc_idx]) * 1)
                discordant_pairs += np.sum((predicted_merit_values[partial_ranking_id][ranked_doc_idx + 1:] > predicted_merit_values[partial_ranking_id][ranked_doc_idx]) * 1)
            kendalls_tau += (((concordant_pairs - discordant_pairs) / (0.5 * num_partial_docs * (num_partial_docs - 1))) + 1) / 2
        elif 'choice' in type:
            num_partial_docs = len(predicted_merit_values[partial_ranking_id])
            concordant_pairs, discordant_pairs = 0, 0
            concordant_pairs += np.sum((predicted_merit_values[partial_ranking_id][1:] < predicted_merit_values[partial_ranking_id][0]) * 1)
            discordant_pairs += np.sum((predicted_merit_values[partial_ranking_id][1:] > predicted_merit_values[partial_ranking_id][0]) * 1)
            kendalls_tau += (((concordant_pairs - discordant_pairs) / (num_partial_docs - 1)) + 1) / 2
    kendalls_tau /= len(test_types)
    print('Ranking Prediction: Normalized Kendall\'s tau = %.4f' % kendalls_tau)


def rank_aggregation_prediction(merit_values, dataset_name, unique_aspects):

    kendalls_tau = 0
    for aspect_idx, aspect in enumerate(unique_aspects):
        if dataset_name == 'paper_review' or dataset_name == 'employee_review' or dataset_name == 'hotel_review' or dataset_name == 'hotel_review_large':
            doc_score = np.loadtxt('./data/' + dataset_name + '/cleaned_data/' + aspect + '_ranking.txt', dtype=int)
        elif dataset_name == 'country':
            doc_score = []
            with open('./data/' + dataset_name + '/cleaned_data/' + aspect + '_ranking.txt') as file:
                for row_idx, row in enumerate(file):
                    row = row.split()
                    doc_score.append([int(row[0]),
                                      500 - row_idx])  # here 500 is just a random maximum number as a score to ranking the documents
            doc_score = np.array(doc_score)

        merit_values_one_aspect = np.copy(merit_values[:, aspect_idx + 1])
        merit_values_one_aspect = merit_values_one_aspect[doc_score[:, 0]]

        kendalls_tau_one_aspect = kendalltau(doc_score[:, 1], merit_values_one_aspect)[0]
        kendalls_tau_one_aspect = (kendalls_tau_one_aspect + 1) / 2
        kendalls_tau += kendalls_tau_one_aspect

    kendalls_tau /= len(unique_aspects)
    print('Rank aggregation prediction: Normalized Kendall\'s tau = %.4f' % kendalls_tau)


def evaluate_kendalls_tau_among_aspects(dataset_name, unique_aspects):

    kendalls_tau, pcc = [], []
    for aspect_idx1 in range(len(unique_aspects) - 1):
        for aspect_idx2 in range(aspect_idx1 + 1, len(unique_aspects)):
            if dataset_name == 'paper_review' or dataset_name == 'employee_review' or dataset_name == 'hotel_review' or dataset_name == 'hotel_review_large':
                doc_score1 = np.loadtxt(
                    './data/' + dataset_name + '/cleaned_data/' + unique_aspects[aspect_idx1] + '_ranking.txt',
                    dtype=int)
                doc_score2 = np.loadtxt(
                    './data/' + dataset_name + '/cleaned_data/' + unique_aspects[aspect_idx2] + '_ranking.txt',
                    dtype=int)
            elif dataset_name == 'country':
                doc_score1, doc_score2 = [], []
                with open('./data/' + dataset_name + '/cleaned_data/' + unique_aspects[
                    aspect_idx1] + '_ranking.txt') as file:
                    for row_idx, row in enumerate(file):
                        row = row.split()
                        doc_score1.append([int(row[0]),
                                           500 - row_idx])  # here 500 is just a random maximum number as a score to ranking the documents
                with open('./data/' + dataset_name + '/cleaned_data/' + unique_aspects[
                    aspect_idx2] + '_ranking.txt') as file:
                    for row_idx, row in enumerate(file):
                        row = row.split()
                        doc_score2.append([int(row[0]),
                                           500 - row_idx])  # here 500 is just a random maximum number as a score to ranking the documents
                doc_score1, doc_score2 = np.array(doc_score1), np.array(doc_score2)

            if dataset_name == 'country' or dataset_name == 'paper_review':
                score1, score2 = [], []
                for source_idx1 in range(len(doc_score1)):
                    if doc_score1[source_idx1, 0] in doc_score2[:, 0]:
                        score1.append(doc_score1[source_idx1, 1])
                        score2.append(doc_score2[doc_score2[:, 0].tolist().index(doc_score1[source_idx1, 0]), 1])
                kendalls_tau.append((kendalltau(score1, score2)[0] + 1) / 2)
                pcc.append(pearsonr(score1, score2)[0])
            else:
                score1 = doc_score1[:, 1][np.argsort(doc_score1[:, 0])]
                score2 = doc_score2[:, 1][np.argsort(doc_score2[:, 0])]
                kendalls_tau.append((kendalltau(score1, score2)[0] + 1) / 2)
                pcc.append(pearsonr(score1, score2)[0])

    return kendalls_tau, pcc