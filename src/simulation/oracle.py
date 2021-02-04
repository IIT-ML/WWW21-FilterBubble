# Script to simulate algorithm as oracle recommender

import os
import time
import pickle
import argparse
from datetime import datetime

import csv
from collections import defaultdict
import numpy as np
import pickle
from os import listdir

prototypes = ['bystanders', 'solid liberas', 'oppty democrats', 'disaffected democrats',
              'devout and diverse', 'new era enterprisers', 'market skeptic repub',
              'country first conserv', 'core conserv']

issues = ['abortion', 'environment', 'guns', 'health care', 'immigration', 'LGBTQ', 'racism', 'taxes',
          'technology', 'trade', 'trump impeachment', 'us military', 'us 2020 election', 'welfare']


def entropy(arr, max_one=False):
    """
    arr is an np arr
    """
    arr = arr / arr.sum()
    e = 0
    for p in arr:
        if p > 0:
            e += p*np.log2(p)

    if max_one:
        e /= np.log2(len(arr))

    return -1*e


def detailed_stats(news, probs, K, keyword='source_partisan_score'):

    sorted_keys = sorted(list(news.keys()))

    sum_prob = 0
    sds = 0

    N = len(probs)
    stance_counts = defaultdict(int)
    source_counts = defaultdict(int)
    topic_counts = defaultdict(int)
    for i in range(N):
        sum_prob += probs[i]
        sds += probs[i]*news[sorted_keys[i]][keyword]
        stance_counts[news[sorted_keys[i]]
                      ['source_partisan_score']] += probs[i]
        source_counts[news[sorted_keys[i]]['source']] += probs[i]
        for t in news[sorted_keys[i]]['cls_label']:
            topic_counts[t] += probs[i]

    return sum_prob, sds, stance_counts, source_counts, topic_counts


def aggregate_stats(news, probs, K, keyword='source_partisan_score'):

    sum_prob, sum_doc_stance, stance_counts, source_counts, topic_counts = detailed_stats(
        news, probs, keyword)
    three_way_stance_counts = defaultdict(int)
    for k in stance_counts:
        if k < 0:
            three_way_stance_counts['left'] += stance_counts[k]
        elif k > 0:
            three_way_stance_counts['right'] += stance_counts[k]
        else:
            three_way_stance_counts['center'] += stance_counts[k]

    results = []

    results.append(sum_prob/K)
    results.append(sum_doc_stance/sum_prob)
    results = results + list(map(lambda x: entropy(np.asarray(list(x.values())), True),
                                 [stance_counts, three_way_stance_counts, source_counts, topic_counts]))
    return results


def arraytocsv(arr, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(arr)):
            writer.writerow(arr[i])


def read_user_csv(file_name):
    '''
    Open a user preference encoded in a csv file. 
    Should be a 14 by 5 array. 
    '''
    with open(file_name, newline="") as csvfile:
        data = list(csv.reader(csvfile))

    data = np.array(data, dtype=float)

    data = data.flatten().tolist()

    return data


def click_prob(uv, iv):
    """
    uv - user vector
    iv - item vector - numpy array
    """
    # normalize iv
    iv_n = iv/iv.sum()

    #dp = np.dot(uv, iv)
    dp = np.dot(uv, iv_n)

    epsilon = 10e-5

    if (dp + epsilon) > 1.0:
        vui = 0.99
    else:
        vui = dp  # the mean of the beta distribution

    ita = 0.98  # the mean of the beta
    pui = vui * ita

    return pui


def main(args, output_folder):

    with open(args.article_file, "rb") as handle:
        news = pickle.load(handle)

    sorted_keys = sorted(list(news.keys()))

    N = len(sorted_keys)
    K = 1000
    M = 5000

    shown_results = dict()
    clicked_results = dict()

    show_probs = np.ones(N)*(K/N)

    results = aggregate_stats(news, show_probs, K, 'source_partisan_score')
    shown_results['random'] = results

    proto_file_names = listdir(args.prototype_folder)

    results = {}

    for file in sorted(proto_file_names):

        print(file)

        uv = read_user_csv(args.prototype_folder+file)

        click_probs = []

        # calculate click prob for all news items
        for k in sorted_keys:
            click_probs.append(click_prob(
                uv, np.asarray(news[k]['topical_vector'])))

        click_probs = np.asarray(click_probs)

        results[file] = aggregate_stats(
            news, show_probs*click_probs, K, 'source_partisan_score')

    clicked_results['random'] = results

    # weight manipulation
    for w in range(10):

        print(w)

        shown_oracle_results = dict()
        clicked_oracle_results = dict()

        for file in sorted(proto_file_names):

            print(file)

            uv = read_user_csv(args.prototype_folder+file)

            click_probs = []

            # calculate click prob for all news items
            for k in sorted_keys:
                click_probs.append(click_prob(
                    uv, np.asarray(news[k]['topical_vector'])))

            click_probs = np.asarray(click_probs)

            weights = np.exp(w*click_probs)

            weights = weights / weights.sum()

            show_probs = np.zeros(N)

            for _ in range(M):
                sample = np.random.choice(N, size=K, replace=False, p=weights)
                show_probs[sample] += 1

            show_probs /= M

            shown_oracle_results[file] = aggregate_stats(
                news, show_probs, K, 'source_partisan_score')
            clicked_oracle_results[file] = aggregate_stats(
                news, show_probs*click_probs, K, 'source_partisan_score')

        shown_results['oracle-w='+str(w)] = shown_oracle_results
        clicked_results['oracle-w='+str(w)] = clicked_oracle_results

    csv_output_file_name = args.output_folder + 'oracle.csv'

    with open(csv_output_file_name, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        # CTR
        writer.writerow(['CTR'] + sorted(clicked_results.keys()))
        # print(sorted(clicked_results.keys()))
        for file in sorted(proto_file_names):
            arr = [file[:-4]]
            for k in sorted(clicked_results.keys()):
                arr.append(clicked_results[k][file][0])
            #print(file, arr)
            writer.writerow(arr)

        metrics = ['Ave Doc Stance', '5-way Stance Entropy',
                   '3-way Stance Entropy', 'Source Entropy', 'Topic Entropy']

        for i in range(len(metrics)):

            writer.writerow([])

            writer.writerow([metrics[i]] + sorted(shown_results.keys()))

            for file in sorted(proto_file_names):
                arr = [file[:-4]]
                for k in sorted(shown_results.keys()):
                    if k == 'random':
                        arr.append(shown_results[k][i+1])
                    else:
                        arr.append(shown_results[k][file][i+1])

                writer.writerow(arr)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parameters of simulation script")

    parser.add_argument(
        "--dataset_type",
        default="balance",
        help="the type of dataset")

    parser.add_argument(
        "--article_file",
        default="../data/balanced_political_news_40k_v2.pkl",
        help="the dataset of political articles",
    )

    parser.add_argument(
        "--prototype_folder",
        default="../data/synthetic_user/",
        help="the profile of 9 prototypes",
    )

    parser.add_argument(
        "--output_folder",
        default="./2021_0130_Expmts/",
        help="the output folder of this experiment",
    )

    args = parser.parse_args()

    with open(args.article_file, "rb") as handle:
        candidate_news = pickle.load(handle)

    now = datetime.now()
    print("now =", now)

    output_folder = (
        args.output_folder
        + "oracle_"
        + args.dataset_type
        + "_"
        + time.strftime("%Y%m%d-%H%M%S")
        + "/"
    )
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    main(args, output_folder)
