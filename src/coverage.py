import itertools
from tqdm import tqdm
import bisect
import pandas as pd
import time
from jpype import *


def get_topk_mups(mups, all_occurences, topk):
    """Retrieve the top k MUPs from the dataset based on their highest abundance.
    Abundance is determined by the sum of the frequency count of all parents of a MUP.

    Args:
        mups (list): list of mups
        all_occurences (dict): dict with all occurences of mups
        topk (int): number of mups to return

    Returns:
        list: top k mups with highest abundance
    """
    mups_abundance_parents = {
        mup: get_max_abundance_in_parents(mup, all_occurences) for mup in mups
    }
    mups_abundance_parents = sorted(
        mups_abundance_parents.items(), key=lambda x: x[1], reverse=True
    )
    return [mup for mup, _ in mups_abundance_parents[:topk]]


def get_max_abundance_in_parents(pattern, all_occurences):
    """Get the maximum frequency among all parents of a pattern.

    Args:
        pattern (tuple): pattern
        all_occurences (dict): dict with frequency counts for all patterns in the dataset

    Returns:
        int: maximum frequency among all parents of a pattern
    """
    parents = gen_parents(pattern)
    parent_occurences = []
    for parent in parents:
        parent_occurences.append(all_occurences[str(parent)])
    return sum(parent_occurences)


def get_occurences_of_parent_patterns(pattern, all_occurences):
    """Generate the parents of a pattern and return all parents
    as well as their frequency counts.

    Args:
        pattern (tuple): pattern
        all_occurences (dict): dict with frequency counts for all patterns in the dataset

    Returns:
        dict: frequency counts of all parents of a pattern
    """
    parent_occurences = {}
    mup_combinations = []
    max_level = len(pattern)  # if len(pattern) < 3 else 3
    for i in range(1, max_level):
        mup_combinations.extend(list(itertools.combinations(pattern, i)))
    # get occurences of all combinations
    for combination in mup_combinations:
        parent_occurences[combination] = all_occurences[str(combination)]
    return parent_occurences


def get_rare_attributes(categories, frequency_count, no_rows, threshold=0.01):
    """Get all rare attribute-values in the dataset.
    Rare attribute-values are those that occur less than the given threshold.

    Args:
        categories (list): list of lists of unique attribute-values
        frequency_count (dict): dict with frequency counts for all patterns in the dataset
        no_rows (int): number of rows in the dataset
        threshold (float, optional): threshold for rare attributes. Defaults to 0.01.

    Returns:
        dict: dict with rare attribute-values and their frequency count and percentage
    """
    threshold = threshold * no_rows
    rare = {
        key: [
            frequency_count[str((key,))],
            frequency_count[str((key,))] / no_rows * 100,
        ]
        for key in sum(categories, [])
        if frequency_count[str((key,))] < threshold
    }
    # sort rare attributes by frequency count and keep dict structure
    rare = dict(sorted(rare.items(), key=lambda x: x[1][0], reverse=False))
    return rare


def get_level_mups(mups, i):
    """Get all MUPs of a certain level.

    Args:
        mups (list): list of mups
        i (int): level

    Returns:
        list: list of mups of level i
    """
    return [mup for mup in mups if len(mup) == i]


def combinatorial_sum(numbers):
    """Calculates the sum of all possible combinations of a list of cardinalities.
    Calculated by creating the cartesian product of all combinations of the cardinalities including the null value.

    Example:
    >>> combinatorial_sum([1, 2, 3])
    20

    Args:
        numbers (list): list of cardinalities

    Returns:
        int: sum of all possible combinations of the cardinalities
    """
    # result = 0
    # n = len(numbers)

    # for r in range(1, n + 1):
    #     for combo in itertools.combinations(numbers, r):
    #         product = 1
    #         for num in combo:
    #             product *= num
    #         result += product

    # return result
    result = 1
    for num in numbers:
        result *= num + 1  # +1 for the null value
    return result


## BASELINE COVERAGE OF ALL UNCOVERED COMBINATIONS ##
### ! USED IN COMPARISON ###
def baseline_coverage_with_keys_all_combs(
    keys_dict, categories, _freq_count, max_level=None, threshold=1
):
    """Computes all uncovered combinations of the dataset. Uncovered combinations are those that are not in the dataset.
    The calculation is based on all known keys of the dataset and the frequency count of all patterns in the dataset.
    Does not perform any pruning or search and always considers all possible combinations.
    Thus, its is only fast for small datasets and a threshold of 1.

    Args:
        keys_dict (dict): dict with all keys of the dataset
        categories (list): list of lists of unique attribute-values
        _freq_count (dict): dict with frequency counts for all patterns in the dataset
        max_level (int, optional): maximum level of the patterns. Defaults to None.
        threshold (int, optional): threshold for a pattern to be considered uncovered. Defaults to 1.

    Returns:
        dict: dict with all uncovered combinations of the dataset
    """
    if max_level is None:
        max_level = len(categories)
    unseen = {}

    if threshold > 1:
        unseen_level1 = {
            k for k in keys_dict.get(1, set()) if _freq_count[str(k)] < threshold
        }
        unseen.setdefault(1, set()).update(unseen_level1)

    for level in tqdm(range(2, max_level + 1)):
        for comb in itertools.combinations(categories, level):
            # get keys of level
            keys_level = keys_dict.get(level, set())
            if threshold > 1:
                keys_level = {k for k in keys_level if _freq_count[str(k)] >= threshold}
            # get all possible combinations of length level from categories and remove those that are already in keys_level
            unseen_level_comb = set(itertools.product(*comb)) - keys_level
            unseen.setdefault(level, set()).update(unseen_level_comb)

    return unseen


## BASELINE COVERAGE OF MAXIMAL UNCOVERED PATTERNS ##
def baseline_coverage_with_keys_searchbased(
    keys_dict, categories, _freq_count, max_level=None, threshold=1
):
    """Computes all uncovered combinations of the dataset. Uncovered combinations are those that are not in the dataset.
    The calculation is based on all known keys of the dataset and the frequency count of all patterns in the dataset.
    Performs pruning and search to reduce the number of combinations that need to be considered. It does so by removing all combinations
    that are parents of a combination that is already in the dataset.
    Thus, it considers only the maximal uncovered patterns of the dataset.
    It is not used in the thesis as it is too slow.

    Args:
        keys_dict (dict): dict with all keys of the dataset
        categories (list): list of lists of unique attribute-values
        _freq_count (dict): dict with frequency counts for all patterns in the dataset
        max_level (int, optional): maximum level of the patterns. Defaults to None.
        threshold (int, optional): threshold for a pattern to be considered uncovered. Defaults to 1.

    Returns:
        dict: dict with all uncovered combinations of the dataset
    """
    if max_level is None:
        max_level = len(categories)
    unseen = {}
    skip = set()
    visited_nodes = 0

    if threshold > 1:
        unseen_level1 = {
            k for k in keys_dict.get(1, set()) if _freq_count[str(k)] < threshold
        }
        skip.update(unseen_level1)
        unseen.setdefault(1, set()).update(unseen_level1)

    for level in tqdm(range(2, max_level + 1)):
        for comb in itertools.combinations(categories, level):
            for u_comb in itertools.product(*comb):
                visited_nodes += 1
                u_comb_parents = gen_parents(u_comb)
                if any(p in skip for p in u_comb_parents):
                    continue
                if _freq_count[str(u_comb)] < threshold and all(
                    _freq_count[str(p)] >= threshold for p in u_comb_parents
                ):
                    skip.add(u_comb)
                    unseen.setdefault(level, set()).add(u_comb)

    print("Visited nodes: ", visited_nodes)

    del skip
    return unseen


###### FREQWALK: ADAPTATION OF P WALK TO KNOWLEDGE OF KEYS AND FREQUENCY COUNT #####


def gen_children_rule1(pattern, cats, freq_count) -> []:
    """Generate only those children of the next level that are on the right side of the pattern.
    BFS traversal of the pattern graph to ensure that each MUP candidate is generated exactly once.
    -> replacing the non-deterministic elements in the right-hand side of its right-most deterministic
    element with an attribute value

    Return a list of tuples of frequency count and attribute-value combinations that are children of the given pattern

    Args:
        pattern (tuple): pattern
        cats (dict): dict with all attribute-values and their index
        freq_count (dict): dict with frequency counts for all patterns in the dataset

    Returns:
        list: list of tuples of frequency count and attribute-value combinations that are children of the given pattern
    """
    pattern_i_rightmost = cats[pattern[-1]]
    cat_list = [
        cat for cat, i in cats.items() if i > pattern_i_rightmost
    ]  # get all categories that are after the rightmost element of pattern

    children = []
    for cat in cat_list:
        p = pattern + (cat,)
        children.append((freq_count[str(p)], p))

    return children


def gen_children_rule1_nodes(pattern, cats) -> []:
    """Generate only those children of the next level that are on the right side of the pattern.
    Return only the attribute-value combinations that are children of the given pattern

    Args:
        pattern (tuple): pattern
        cats (dict): dict with all attribute-values and their index

    Returns:
        list: list of attribute-value combinations that are children of the given pattern
    """
    pattern_i_rightmost = cats[pattern[-1]]
    cat_list = [
        cat for cat, i in cats.items() if i > pattern_i_rightmost
    ]  # get all categories that are after the rightmost element of pattern

    return [pattern + (cat,) for cat in cat_list]


def gen_parents(pattern):
    """Generate all parents of a given pattern of the previous level.
    Does so by removing one element from the pattern at a time.

    Args:
        pattern (tuple): pattern

    Returns:
        list: list of all parents of the pattern
    """
    parents = [p for p in itertools.combinations(pattern, len(pattern) - 1) if p]
    return parents


##EDITED TO USE APRIORI PRUNING AND SEARCH!!
### ! USED IN COMPARISON ###
def freqwalk_frequency_weight(categories, _freq_count, threshold=1):
    """Computes Maximal Uncovered Patterns (MUPs) of the dataset. MUPs are those patterns that are not (sufficiently) in the dataset and have no uncovered parents.
    The calculation is based on the frequency count of all patterns in the dataset.
    See thesis for pseudocode and further explanation.

    Args:
        categories (list): list of lists of unique attribute-values
        _freq_count (dict): dict with frequency counts for all patterns in the dataset
        threshold (int, optional): threshold for a pattern to be considered uncovered. Defaults to 1.

    Returns:
        dict: dict with all MUPs of the dataset
    """
    cats = {cat: i for i, cats in enumerate(categories) for cat in cats}
    mups_dict = {}
    neighbors = []
    for i in sum(categories, []):
        bisect.insort(neighbors, (_freq_count[str((i,))], (i,)))

    while neighbors:
        p_freq, p_node = neighbors.pop(0)
        p_node_parents = gen_parents(p_node)
        mups_level = mups_dict.get(len(p_node) - 1, set())
        if any(p in mups_level for p in p_node_parents):
            continue
        if p_freq >= threshold:
            children = gen_children_rule1(p_node, cats, _freq_count)
            for c in children:
                bisect.insort(neighbors, c)
        elif p_freq < threshold and all(
            _freq_count[str(p)] >= threshold for p in p_node_parents
        ):
            mups_dict.setdefault(len(p_node), set()).add(p_node)
            # remove all children of p_node from neighbors
            children = gen_children_rule1_nodes(p_node, cats)
            neighbors = [n for n in neighbors if n[1] not in children]

    return mups_dict


#### COVERAGEJAVA DEEPDIVER ALGORITHM VIA JPYPE ####
# Asudeh, Abolfazl, et al. “Assessing and Remedying Coverage for a Given Dataset.” 2019 IEEE 35th International Conference on Data Engineering (ICDE), Apr. 2019. Crossref, https://doi.org/10.1109/icde.2019.00056. #
### ! USED IN COMPARISON ###
def get_coverage_java(threshold, df):
    """Python Wrapper for the Java implementation of the DeepDiver algorithm.
    DeepDiver by Asudeh et al. is a search-based algorithm that finds all maximal uncovered patterns (MUPs) of a dataset.

    Args:
        threshold (int): threshold for a pattern to be considered uncovered
        df (pandas.DataFrame): dataset

    Returns:
        t: time needed to find all MUPs
        mups: list of all MUPs
    """
    dff = df.dtypes

    # get classes
    cpopt = "-Djava.class.path=%s" % ("CoverageJava/target/classes/")
    if not isJVMStarted():
        startJVM(getDefaultJVMPath(), "-ea", cpopt)

    dataset = JClass("io.DataSet")
    hybrid = JClass("search.HybridSearch")

    valid_cols, valid_col_indices, cardinality, categories = [], [], [], []

    # grouping numerical
    for col in list(df):
        if dff[col] != object:
            cur_col = list(df[col])
            dimension = len(set(cur_col))
            if dimension <= 10:
                df[col] = str(df[col])
            else:
                df[col] = [str(bucket) for bucket in pd.cut(cur_col, dimension)]

    for i, col in enumerate(list(df)):
        # restrict cardinality
        temp_set = set(list(df[col]))  # unique values

        if len(temp_set) <= 10 and len(temp_set) >= 2:
            valid_cols.append(col)
            valid_col_indices.append(i)
            cardinality.append(len(temp_set))

            # encoding valid categorical columns as numeric (one-hot encoding)
            labels, uniques = pd.factorize(list(df[col]), sort=True)
            df[col] = labels
            categories.append([col + ":" + str(unique) for unique in uniques])

    temp = df[valid_cols].astype(str)
    temp.to_csv("temp.csv", index=False)

    t_ = time.time()
    dataset1 = dataset(
        "temp.csv", cardinality, [i for i in range(len(valid_cols))], temp.shape[0]
    )

    hybrid1 = hybrid(dataset1)
    a = hybrid1.findMaxUncoveredPatternSet(threshold)  # threshold, maxLevel

    mups = [i.getStr() for i in a]
    t = time.time() - t_
    print("time: ", t)
    print("mups: ", len(mups))

    # get all children patterns of mups to get total amount of uncovered patterns
    # fill in list of uncovered patterns with children patterns (max combination: 3 as three columns) -> add to list all patterns of only one or two elements with all other combinations of length 3
    # for pattern in a:
    #     children_pattern = dataset1.getAllChildren(pattern)
    #     # print("children pattern: ", children_pattern)
    #     mups.extend([i.getStr() for i in children_pattern])

    return t, mups
