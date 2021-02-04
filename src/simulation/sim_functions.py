"""
This script contans all recommender related functions
"""

import csv
import time

import pickle
import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import SGDClassifier

from user_choice import user_interaction, random_select, simple_doct_product


def get_issue_mapping(candidate_news):
    '''
    Get a mapping object from issue to score to pks
    To bootstrap
    '''
    issue_mapping = {}

    for key, value in candidate_news.items():

        partisan_score = int(value["source_partisan_score"])

        for iss in value["cls_label"]:

            if iss not in issue_mapping:
                issue_mapping[iss] = {}

            if partisan_score not in issue_mapping[iss]:
                issue_mapping[iss][partisan_score] = [key]
            else:
                issue_mapping[iss][partisan_score].append(key)

    return issue_mapping


def build_train(candidate_news, pos_pks, neg_pks, weights_map, return_indices=False):
    '''
    Build X, y for training 
    '''
    x = []
    y = []
    w = []
    indices = []

    for pk in pos_pks:

        x.append(candidate_news[pk]["feature_vector"])
        y.append(1)
        w.append(weights_map[candidate_news[pk]['source_partisan_score']])
        indices.append(pk)

    for pk in neg_pks:

        x.append(candidate_news[pk]["feature_vector"])
        y.append(0)
        w.append(weights_map[candidate_news[pk]['source_partisan_score']])
        indices.append(pk)

    x = np.array(x)

    if return_indices:
        return x, y, w, indices

    return x, y, w


def build_test(candidate_news, pk_pool):
    '''
    Build X, pks for testing
    '''
    x = []
    indices = []

    for pk in pk_pool:

        x.append(candidate_news[pk]["feature_vector"])
        indices.append(pk)

    x = np.array(x)

    return x, indices


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


def random_bootstrap(user_vector, issue_mapping, candidate_news, budget, seed=42):
    '''
    Randomly choose the equal number of articles from each topic
    '''
    chosen_topic = [
        "abortion",
        "environment",
        "guns",
        "health care",
        "immigration",
        "LGBTQ",
        "racism",
        "taxes",
        "technology",
        "trade",
        "trump impeachment",
        "us military",
        "us 2020 election",
        "welfare",
    ]

    l_topic = len(chosen_topic)

    size_list = [budget // l_topic for i in range(l_topic - 1)]
    size_list.append(budget - np.sum(size_list))

    topic_mapping = {}

    for topic, values in issue_mapping.items():

        topic_pk_list = []

        for lean, pks in values.items():
            topic_pk_list.extend(pks)

        topic_mapping[topic] = topic_pk_list

    pos_pks = []
    neg_pks = []

    overall = set()

    for size, topic in zip(size_list, chosen_topic):

        bucket = set()

        candidate_pks = topic_mapping[topic]
        np.random.shuffle(candidate_pks)

        while len(bucket) < size:

            pk = candidate_pks.pop()

            if pk not in overall:
                s = user_interaction(user_vector, candidate_news[pk])
                if s:
                    pos_pks.append(pk)
                else:
                    neg_pks.append(pk)

                bucket.add(pk)

        overall.add(pk)

    return pos_pks, neg_pks


def random_bootstrap_v2(user_vector, issue_mapping, candidate_news, budget, seed=42):
    '''
    Randomly choose budget articles
    '''
    pos_pks = []
    neg_pks = []

    pks = list(candidate_news.keys())
    np.random.shuffle(pks)

    for pk in pks[:budget]:

        s = user_interaction(user_vector, candidate_news[pk])
        if s:
            pos_pks.append(pk)
        else:
            neg_pks.append(pk)

    return pos_pks, neg_pks


def cbnf_algorithm(user_vector, candidate_news, ran_idx, partisan_weights, randomness=0.0, bootstrap=700, num_rec=1000):
    '''
    content-based algorithm
    '''
    np.random.seed(ran_idx)

    history = {}
    issue_mapping = get_issue_mapping(candidate_news)

    pos_pks, neg_pks = random_bootstrap(
        user_vector, issue_mapping, candidate_news, bootstrap
    )

    unlabeled_pool = set(candidate_news.keys())
    unlabeled_pool.difference_update(pos_pks)
    unlabeled_pool.difference_update(neg_pks)

    history["bootstrap_pos"] = pos_pks.copy()
    history["bootstrap_neg"] = neg_pks.copy()
    history["clicked"] = []
    history["shown"] = []

    x_train, y_train, w_train = build_train(
        candidate_news, pos_pks, neg_pks, partisan_weights)
    x_test, x_test_pk = build_test(candidate_news, unlabeled_pool)

    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train)

    class_weights = {0: class_weights[0], 1: class_weights[1]}

    cls = SGDClassifier(loss="log", class_weight=class_weights)
    cls.fit(x_train, y_train, sample_weight=w_train)

    # Start to recommend
    for _ in range(num_rec):

        if np.random.rand() < randomness:
            top_news_idx = np.random.randint(0, len(x_test_pk))
        else:
            y_prob = cls.predict_proba(x_test)[:, 1]
            top_news_idx = np.argsort(y_prob)[::-1][0]

        top_news_pk = x_test_pk[top_news_idx]
        top_news = candidate_news[x_test_pk[top_news_idx]]

        unlabeled_pool.remove(top_news_pk)

        flag = user_interaction(user_vector, top_news, ranked=True)

        newx = [candidate_news[top_news_pk]["feature_vector"]]
        newx = np.array(newx)

        if flag:
            newy = [1]
            pos_pks.append(top_news_pk)
            history["clicked"].append(top_news)
        else:
            newy = [0]
            neg_pks.append(top_news_pk)

        x_test = np.delete(x_test, top_news_idx, 0)
        x_test_pk.pop(top_news_idx)

        neww = [partisan_weights[candidate_news[top_news_pk]
                                 ['source_partisan_score']]]

        cls.partial_fit(newx, newy, sample_weight=neww)

        history["shown"].append(top_news)

    history['clf'] = cls

    return history


def alg_cbnf(user_vec, candidate_news, output_file, args):
    '''
    wrapper of content-based algorithm
    '''
    rst = {}

    for ite in range(args.num_runs):
        start = time.time()
        history = cbnf_algorithm(
            user_vec, candidate_news, ite, args.partisan_weights, args.randomness
        )
        rst[ite] = history
        end = time.time()
        print(
            "This is %s-th run of %s, spending %s seconds"
            % (str(ite), output_file.split(".")[0], str(end - start))
        )

    with open(output_file, "wb") as handle:
        pickle.dump(rst, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("The simulation of this run is complete!")

    return rst


def highest(user, candidate_news, topn=1000):
    '''
    Pick the top n articles with the hightest score. 
    Used for oracle algorithm
    '''
    scores = []
    pks = []

    for key, value in candidate_news.items():

        article_vector = value["topical_vector"]

        s = simple_doct_product(user, article_vector)

        scores.append(s)
        pks.append(key)

    topn_idx = np.argsort(scores)[::-1][:topn]

    topn_pks = [pks[i] for i in topn_idx]

    return topn_pks


def oracle_algorithm(user, candidate_news, num_timestep=1000, top_N=1):
    '''
    Oracle algorithm
    '''
    history = {}
    history["clicked"] = []
    history["shown"] = []

    candidate_pks = set(list(candidate_news.keys()))

    # In here, it is gonna calculate the top ones at one time.
    # Then sorted by descending order.

    topn_pks = highest(user, candidate_news, num_timestep)

    for num_ite in range(num_timestep):

        top_N_pk = topn_pks[num_ite]

        selected_news = candidate_news[top_N_pk]

        flag = user_interaction(user, selected_news, ranked=False)

        if flag:
            history["clicked"].append(selected_news)

        history["shown"].append(selected_news)

        candidate_pks.remove(selected_news["article_id"])

    return history


def alg_oracle(user_vec, candidate_news, output_file, args):
    '''
    Wrapper of oracle algorithm
    '''
    rst = {}

    for ite in range(args.num_runs):

        history = oracle_algorithm(user_vec, candidate_news)
        rst[ite] = history

    with open(output_file, "wb") as handle:
        pickle.dump(rst, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("The simulation of %s is complete!" %
          output_file.split("/")[-1].split(".")[0])
    print()

    return rst


def random_algorithm(user, candidate_news, num_timestep=1000, top_N=1):
    '''
    Random algorithm
    '''
    history = {}
    history["clicked"] = []
    history["shown"] = []

    candidate_pks = set(list(candidate_news.keys()))

    for _ in range(num_timestep):

        top_N_pks = random_select(candidate_pks, top_N)

        selected_news = candidate_news[top_N_pks[0]]

        flag = user_interaction(user, selected_news, ranked=False)

        if flag:
            history["clicked"].append(selected_news)

        history["shown"].append(selected_news)

        candidate_pks.remove(selected_news["article_id"])

    return history


def alg_random(user_vec, candidate_news, output_file, args):
    '''
    Wrapper of random algorithm
    '''
    rst = {}

    for ite in range(args.num_runs):

        history = random_algorithm(user_vec, candidate_news)
        rst[ite] = history

    with open(output_file, "wb") as handle:
        pickle.dump(rst, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("The simulation of %s is complete!" %
          output_file.split("/")[-1].split(".")[0])
    print()

    return rst
