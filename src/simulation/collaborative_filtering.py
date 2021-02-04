import os
import time
import pickle
import argparse
import numpy as np

from datetime import datetime

from user_choice import user_interaction
from user_generation import generation
from sim_functions import get_issue_mapping, random_bootstrap

from sklearn.decomposition import non_negative_factorization

import warnings
warnings.filterwarnings("ignore")


prototypes = ['bystanders', 'solid liberas', 'oppty democrats', 'disaffected democrats',
              'devout and diverse', 'new era enterprisers', 'market skeptic repub',
              'country first conserv', 'core conserv']

issues = ['abortion', 'environment', 'guns', 'health care', 'immigration', 'LGBTQ', 'racism', 'taxes',
          'technology', 'trade', 'trump impeachment', 'us military', 'us 2020 election', 'welfare']


def read_user_pkl(data):
    '''
    convert generated users to a dict object
    '''
    res = {}
    idx = 0

    for prototype, values in data.items():

        for matrix in values:

            res[idx] = {}
            res[idx]['ptt'] = prototype
            res[idx]['vec'] = matrix.flatten().tolist()
            idx += 1

    return res


def cf_algorithm(users, candidate_news, random_seed, randomness, hidden_dim=40, bootstrap_size=700, num_timestep=1000):
    '''
    collaborative-filtering algorithm
    '''
    users_clicked_pool = {}

    issue_mapping = get_issue_mapping(candidate_news)

    # Create a dict from pk to column index
    # and create a dict from column index to pk
    pk2col = {}
    col2pk = {}
    for idx, pk in enumerate(list(candidate_news.keys())):
        pk2col[pk] = idx
        col2pk[idx] = pk

    history = {}
    for idx in range(len(users)):
        history[idx] = {}
        history[idx]['prototype'] = users[idx]['ptt']
        history[idx]['shown'] = []
        history[idx]['clicked'] = []

    matrix = np.zeros((len(users), len(candidate_news)))

    # Initialize the bootstrap for each user
    for idx, user in sorted(users.items()):

        users_clicked_pool[idx] = set()

        pos_pks, neg_pks = random_bootstrap(
            user['vec'], issue_mapping, candidate_news, bootstrap_size)

        users_clicked_pool[idx].update(pos_pks)
        users_clicked_pool[idx].update(neg_pks)

        for pk in pos_pks:
            matrix[idx, pk2col[pk]] = 1

    # Start to recommend, first test nmf
    # Add randomness parameters in here
    for _ in range(num_timestep):

        W, H, _ = non_negative_factorization(
            matrix, n_components=hidden_dim, init='random', random_state=random_seed, max_iter=250)
        new_matrix = np.matmul(W, H)

        for idx, row in enumerate(new_matrix):

            user_vector = users[idx]['vec']

            if np.random.rand() < randomness:

                keys = list(candidate_news.keys())
                np.random.shuffle(keys)

                for pk in keys:
                    if pk not in users_clicked_pool[idx]:
                        candidate_pk = pk
                        break
            else:
                indices = np.argsort(row)[::-1]

                for col_idx in indices:
                    if col2pk[col_idx] not in users_clicked_pool[idx]:
                        candidate_pk = col2pk[col_idx]
                        break

            # Remove from the pool
            users_clicked_pool[idx].add(candidate_pk)

            top_news = candidate_news[candidate_pk]
            flag = user_interaction(user_vector, top_news, ranked=True)

            top_col_idx = pk2col[candidate_pk]

            if flag:
                matrix[idx, top_col_idx] = 1
                history[idx]['clicked'].append(top_news)
            else:
                assert matrix[idx, top_col_idx] == 0

            history[idx]['shown'].append(top_news)

    return history


def alg_cf(args, users, num_runs, candidate_news, output_file):
    '''
    wrapper of collaborative-filtering algorithm
    '''
    rst = {}

    for ite in range(num_runs):

        start = time.time()
        history = cf_algorithm(users, candidate_news, ite,
                               args.randomness, num_timestep=args.num_recommends)
        history['users'] = users

        rst[ite] = history
        end = time.time()
        print("This is %s-th run, spending %s seconds" %
              (str(ite), str(end-start)))

    with open(output_file, 'wb') as handle:
        pickle.dump(rst, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("The simulation of this run is complete!")

    return rst


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parameters of simulation script")

    parser.add_argument(
        "--num_runs",
        default=1,
        help="the number of runs"
    )

    parser.add_argument(
        "--num_people",
        default=900,
        help="the number of people per run"
    )

    parser.add_argument(
        "--randomness",
        default=0.0,
        help="the randomness parameter coupled with CF algorithm"
    )

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
        "--pew_study",
        default="../data/pew0423.csv",
        help="the file of pew study to generate prototypes",
    )

    parser.add_argument(
        "--num_recommends",
        default=1000,
        help="the number of recommendations for each user",
    )

    parser.add_argument(
        "--users",
        default="../data/900.pkl",
        help="The path of users file",
    )

    parser.add_argument(
        "--output_folder",
        default="../",
        help="the output folder of this experiment",
    )

    parser.add_argument(
        "--output_key",
        default="shown",
        help="the keyword to select the output to analyze. Could be clicked or shown",
    )

    args = parser.parse_args()

    with open(args.article_file, "rb") as handle:
        candidate_news = pickle.load(handle)

    if args.users:
        print("Loading the users' profile")
        with open(args.users, "rb") as f:
            users = pickle.load(f)

    else:
        record = generation(args.num_people, args.pew_study, equal=True)
        users = read_user_pkl(record)

        file_name = "./data/" + str(args.num_people) + ".pkl"

        with open(file_name, "wb") as f:
            pickle.dump(users, f)

    now = datetime.now()
    print("now =", now)

    output_folder = (
        args.output_folder
        + "CF_"
        + args.dataset_type
        + "_"
        + time.strftime("%Y%m%d-%H%M%S")
        + "_R="
        + str(args.randomness)
        + "/"
    )

    output_file = output_folder + "cfmf.pkl"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    record = alg_cf(args, users, args.num_runs, candidate_news, output_file)

    prototype_record = {}

    for synth in prototypes:
        prototype_record[synth] = {}

    for key, value in record.items():
        for idx in range(args.num_people):
            prototype = value[idx]['prototype']
            length = len(prototype_record[prototype])
            prototype_record[prototype][length] = value[idx]

    for key, value in prototype_record.items():
        file_name = output_folder + key + ".pkl"
        with open(file_name, "wb") as file_handle:
            pickle.dump(value, file_handle)
