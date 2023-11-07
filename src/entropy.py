# Description: Functions for calculating entropy and related quantities.
import numpy as np
import matplotlib.pyplot as plt
import itertools
import streamlit as st

from scipy.spatial.distance import jensenshannon


def calc_hill_numbers(data, q_range):
    """
    Calculate Hill numbers for a given dataset and diversity orders in q_range.

    Arguments:
    - data: A list containing the species abundance data.
    - q_range: The range of diversity orders (positive real numbers).

    Returns:
    - A list of Hill numbers.
    """
    total_abundance = np.sum(data)

    def normalize_row(count):
        return count / total_abundance

    data = [d for d in data if d > 0]  # remove zero counts
    normalized_data = list(map(normalize_row, data))
    num_species = len(normalized_data)

    hill_numbers = []
    for q in q_range:
        hill_number = 0
        temp_hill = 0
        for p in normalized_data:
            if q == 1:
                temp_hill += p * np.log(p)
            else:
                temp_hill += p**q
        if q == 1:
            hill_number = np.exp(-temp_hill)
        else:
            hill_number = temp_hill ** (1 / (1 - q))

        hill_numbers.append(hill_number)

    return hill_numbers


def plot_hill_numbers(data, q_range, labels=None):
    """
    Plot Hill numbers for a given dataset and diversity orders in q_range.

    Arguments:
    - data: A list of lists containing the species abundance data for different categories (sites).
    - q_range: The range of diversity orders (positive real numbers).
    """
    plt.rcParams["figure.figsize"] = (5, 4)

    for d in range(len(data)):
        # print(data[d])
        hill_numbers = calc_hill_numbers(data[d], q_range)
        if labels:
            plt.plot(q_range, hill_numbers, label=labels)
        # else:
        #     plt.plot(q_range, hill_numbers, label="Category " + str(d))

    plt.xlabel("q")
    plt.ylabel("Hill number")
    plt.legend()
    plt.show()


def get_uncovered_occurences(uncovered_p, occurences_general):
    """Get occurences of uncovered patterns.

    Args:
        uncovered_p: list of uncovered patterns
        occurences_general: dict of occurences of all patterns

    Returns:
        uncovered_occurences: dict of occurences of uncovered patterns
    """
    uncovered_occurences = {}
    for p in uncovered_p:
        p_general = tuple([x.split(":")[0] for x in p])
        uncovered_occurences[p_general] = occurences_general[p_general]
    return uncovered_occurences


# for this, calculate the aggregate Jensen-Shannon (JS) divergence between the probability vector of pattern group and uniform distribution
# https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
def get_js_distance(x):
    """Get Jensen-Shannon distance between a given distribution and uniform distribution.

    Args:
        x: list of counts

    Returns:
        Jensen-Shannon distance
    """
    if len(x) == 0:
        return 0
    # uniform distribution
    q = [1 / len(x)] * len(x)
    # probability vector of pattern group (assumes x is a list of counts)
    p = [i / sum(x) for i in x]
    # print(q[0], p)
    return jensenshannon(p, q, keepdims=False, base=2)


def get_true_shannon_entropy(distribution):
    """Get Shannon entropy for a given distribution.

    Args:
        distribution: list of counts

    Returns:
        Shannon entropy
    """
    p = [i / sum(distribution) for i in distribution]
    return sum([-i * np.log2(i) for i in p])


def get_categorical_true_diversity(df, pattern_count, col1, col2, categories):
    """Get true diversity for each category in col1. True diversity is calculated for each category in col1
    by considering the distribution of patterns in col2.

    Args:
        df: dataframe
        pattern_count: dict of pattern counts
        col1: attribute name to calculate true diversity for
        col2: attribute name to calculate true diversity against
        categories: dict of categories

    Returns:
        true_divs: dict of true diversity for each category in col1
        shannon_divs: dict of Shannon entropy for each category in col1
        max_even_shannon_divs: dict of Shannon entropy for each category in col1
    """
    true_divs = {}
    shannon_divs = {}
    max_even_shannon_divs = {}
    pos_col1 = df.columns.get_loc(col1)
    pos_col2 = df.columns.get_loc(col2)
    for i in categories[pos_col1]:
        if pos_col1 < pos_col2:
            distribution = [pattern_count[str((i, j))] for j in categories[pos_col2]]
        else:
            distribution = [pattern_count[str((j, i))] for j in categories[pos_col2]]
        true_len_dist = len([x for x in distribution if x > 0])
        max_even_distribution = [1 / true_len_dist] * true_len_dist
        # print(i, distribution)
        t_div = calc_hill_numbers(distribution, list(np.arange(0, 3.1, 0.1)))
        max_even_t_div = calc_hill_numbers(
            max_even_distribution, list(np.arange(0, 3.1, 0.1))
        )
        true_divs[f"{i}->{col2}"] = t_div
        shannon_divs[i] = t_div[10]
        max_even_shannon_divs[i] = max_even_t_div[10] - t_div[10]
    return true_divs, shannon_divs, max_even_shannon_divs


@st.cache_data
def get_dist_from_expected_distribution(occurences_general, expected_distribution=None):
    """Get distance from expected distribution for each pattern group.
    Expected distribution is uniform distribution. If not provided, it is calculated from occurences_general.

    Args:
        occurences_general: dict of occurences of all patterns
        expected_distribution: dict of expected distribution of all patterns

    Returns:
        dist_from_expected: dict of distance from expected distribution for each pattern group
    """
    dist_to_uniform = {}
    for k, v in occurences_general.items():
        dist = get_js_distance(v)
        dist = dist[0] if type(dist) == np.ndarray else dist
        dist_to_uniform[k] = dist

    # sort dist_to_uniform.keys() by value
    dist_to_uniform = {
        k: v
        for k, v in sorted(
            dist_to_uniform.items(), key=lambda item: item[1], reverse=True
        )
    }
    return dist_to_uniform
