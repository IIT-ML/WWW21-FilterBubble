# Script to simulate algorithm as random, content (SGD)

import os
import time
import pickle
import argparse
from datetime import datetime
from multiprocessing import Process

from sim_functions import alg_cbnf, alg_random
from sim_functions import read_user_csv


prototypes = ['bystanders', 'solid liberas', 'oppty democrats', 'disaffected democrats',
              'devout and diverse', 'new era enterprisers', 'market skeptic repub',
              'country first conserv', 'core conserv']

issues = ['abortion', 'environment', 'guns', 'health care', 'immigration', 'LGBTQ', 'racism', 'taxes',
          'technology', 'trade', 'trump impeachment', 'us military', 'us 2020 election', 'welfare']


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parameters of simulation script")

    parser.add_argument("--num_runs", default=100, help="the number of runs")

    parser.add_argument(
        "--algorithm",
        default="content",
        help="the algorithm that is chosen to run"
    )

    parser.add_argument(
        "--randomness",
        default=0.0,
        type=float,
        help="the randomness parameter coupled with content based algorithm"
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
        "--prototype_folder",
        default="../data/synthetic_user/",
        help="the profile of 9 prototypes",
    )

    parser.add_argument(
        "--prototypes",
        default=['devout and diverse',
                 'new era enterprisers', 'market skeptic repub'],
        help="the prototype to run",
    )

    parser.add_argument(
        "--output_folder",
        default="./2021_0130_Expmts/",
        help="the output folder of this experiment",
    )

    parser.add_argument(
        "--output_key",
        default="shown",
        help="the keyword to select the output to analyze. Could be clicked or shown",
    )

    parser.add_argument(
        "--debias_flag",
        default=False,
        help="the partisan weights to debias",
    )

    parser.add_argument(
        "--partisan_weights",
        default={
            -2: 2,
            -1: 0.5,
            0: 1,
            1: 1,
            2: 1
        },
        help="the partisan weights to debias",
    )

    args = parser.parse_args()

    if not args.debias_flag:
        for i in [-2, -1, 0, 1, 2]:
            args.partisan_weights[i] = 1

    algorithm_dict = {
        "random": alg_random,
        "content": alg_cbnf,
    }

    with open(args.article_file, "rb") as handle:
        candidate_news = pickle.load(handle)

    now = datetime.now()
    print("now =", now)

    output_folder = (
        args.output_folder
        + args.algorithm
        + "_"
        + args.dataset_type
        + "_"
        + time.strftime("%Y%m%d-%H%M%S")
        + "_R="
        + str(args.randomness)
        + "/"
    )
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    procs = []

    for prototype in args.prototypes:

        user_file = args.prototype_folder + prototype + ".csv"
        user_vec = read_user_csv(user_file)
        output_file = output_folder + prototype + ".pkl"

        func = algorithm_dict[args.algorithm]
        proc = Process(
            target=func, args=(user_vec, candidate_news, output_file, args)
        )
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
