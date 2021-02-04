"""
The script to generate synthetic users based on the pew study.
The latest pew study file is "pew0423.csv"
"""

import csv
import pickle

import numpy as np


def read_csv_pew(csv_file):
    """
    Read the csv file of pew study
    """
    prototypes = [
        "bystanders",
        "solid liberas",
        "oppty democrats",
        "disaffected democrats",
        "devout and diverse",
        "new era enterprisers",
        "market skeptic repub",
        "country first conserv",
        "core conserv",
    ]

    issues = [
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

    issue_stats = {}
    protp_record = {}

    for pttp in prototypes:
        protp_record[pttp] = {}

    for iss in issues:
        issue_stats[iss] = {}

    with open(csv_file, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):

            if line[0] in issues:

                issue_stats[line[0]]["weighted_ave"] = float(line[-1]) / 100

                scores = line[1:-1]
                scores = [float(s) / 100 for s in scores]
                issue_stats[line[0]]["min"] = min(scores)
                issue_stats[line[0]]["max"] = max(scores)
                issue_stats[line[0]]["std"] = np.std(scores)

            # record protp_record
            if i != 0:
                for s, p in zip(scores, prototypes):
                    protp_record[p][line[0]] = s

    for key, values in issue_stats.items():

        left = values["weighted_ave"] - values["min"]
        right = values["max"] - values["weighted_ave"]

        l_portion = left / 5
        r_portion = right / 5

        ll1 = [
            values["min"] + 2 * l_portion,
            values["min"] + 4 * l_portion,
            values["min"] + 5 * l_portion + r_portion,
            values["min"] + 5 * l_portion + 3 * r_portion,
            1,
        ]

        issue_stats[key]["boundaries"] = ll1

    res = {}

    for proto, values in protp_record.items():

        res[proto] = {}

        for iss in issues:

            bound = issue_stats[iss]["boundaries"]

            percent = values[iss]

            if percent <= bound[0]:

                lean = 2

            elif percent <= bound[1]:

                lean = 1

            elif percent <= bound[2]:

                lean = 0

            elif percent <= bound[3]:

                lean = -1

            else:
                lean = -2

            res[proto][iss] = lean

    return issue_stats, res, protp_record


def amenable_interest(lean, interest, std, dist=1):
    """
    lean is from [-2, -1, 0, 1, 0]
    interest is from 0 to 1
    std is calculated per topic
    """

    epsilon = 10e-3

    idx_list = [-2, -1, 0, 1, 2]

    res = []

    lean_idx = idx_list.index(lean)

    for idx, key in enumerate(idx_list):

        if idx != idx_list:
            s = max(epsilon, interest * (1 - dist * abs(lean_idx - idx) * std))
        else:
            s = interest

        if (s + epsilon) > 1:
            s = 0.999

        res.append(s)

    return res


def get_alpha(score, weighted_ave, C=8):
    """
    Get the main interest score of the lean
    return: score is from 0 to 1
    """

    alpha = 1 + C * abs(score - weighted_ave) / weighted_ave

    return alpha


def generation(num_users, csv_file, dist=1, equal=False, random_number=42):
    """
    Generate synthetic prototype from the pew study numbers
    """
    np.random.seed(random_number)

    percent = [5, 20, 13, 12, 8, 10, 11, 6, 15]
    percent = [i / 100 for i in percent]

    issues = [
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

    prototypes = [
        "bystanders",
        "solid liberas",
        "oppty democrats",
        "disaffected democrats",
        "devout and diverse",
        "new era enterprisers",
        "market skeptic repub",
        "country first conserv",
        "core conserv",
    ]

    issue_stats, protp_lean, protp_record = read_csv_pew(csv_file)

    record = {}

    if equal:
        outcome = [num_users//len(prototypes) for i in prototypes]
    else:
        outcome = np.random.multinomial(num_users, percent).tolist()

    print(outcome)

    for num, prototype in zip(outcome, prototypes):

        record[prototype] = []

        for _ in range(num):

            s_array = []

            for topic in issues:

                score = float(protp_record[prototype][topic])

                plean = protp_lean[prototype][topic]

                alpha = get_alpha(score, issue_stats[topic]["weighted_ave"])

                interest = np.random.beta(alpha, 1)

                t_list = amenable_interest(
                    plean, interest, issue_stats[topic]["std"], dist
                )

                s_array.append(t_list)
            s_array = np.array(s_array)

            record[prototype].append(s_array)

    return record


if __name__ == "__main__":

    csv_file_path = "../data/pew0423.csv"
    record = generation(1000, csv_file_path, 1)

    # This is for the CF algorithm.
    with open("1000users.pkl", "wb") as f:
        pickle.dump(record, f)

    # Average the synthetic users for content-based algorithm
    for key, value in record.items():

        value = np.array(value)

        value_ave = np.mean(value, axis=0)

        print(key, value.shape)

        file_name = "./synthetic_user/" + key + ".csv"

        np.savetxt(file_name, value_ave, delimiter=",")
