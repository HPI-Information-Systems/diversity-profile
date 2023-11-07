import pandas as pd
import numpy as np

import itertools

from bounter import CountMinSketch

from tqdm import tqdm
import streamlit as st


def powerset(iterable):
    """Calculates the powerset of an iterable. The powerset is the set of all subsets of the iterable.
    powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    Args:
        iterable (iterable): The iterable to calculate the powerset for.

    Returns:
        iterable: The powerset of the iterable.
    """
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)
    )


def powerset_maxlength(iterable, max_level):
    """Calculates the powerset of an iterable. The powerset is the set of all subsets of the iterable.
    powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    Args:
        iterable (iterable): The iterable to calculate the powerset for.
        max_level (int): The maximum level of combinations to calculate.

    Returns:
        iterable: The powerset of the iterable.
    """
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, max_level + 1)
    )


def factorize_data(df_raw):
    """Factorizes the data in a dataframe. The factorization is done column-wise.
    The factorization is done by using the pandas.factorize function doing the following:
    For each column, the unique values are extracted and mapped to a unique integer.
    The values of the column are then replaced by the mapped integer.

    Example:
    Input: 0 1 2 3 4 5 6 7 8 9
           a a b c c b a b c a
    Output: 0 1 2 3 3 2 0 2 3 0

    Args:
        df_raw (pd.DataFrame): The dataframe to factorize. Must contain only categorical data.

    Returns:
        df: The factorized dataframe.
        mapping: The mapping from the original values to the factorized values.
    """
    df = df_raw.copy()
    mapping = {}
    for col in tqdm(range(len(df.columns))):
        labels, uniques = pd.factorize(df[df.columns[col]])
        labels_ = [f"{str(col)}:{str(l)}" for l in labels]
        labels_unique = list(np.unique(labels))
        mapping[df.columns[col]] = dict(zip(uniques, labels_unique))
        df[df.columns[col]] = labels_
    del labels, labels_, uniques
    return df, mapping


@st.cache_data
def encode_data(df_raw):
    """Encodes the data in a dataframe. The encoding is done column-wise.
    The encoding follows the following scheme: column:value

    Args:
        df_raw (pd.DataFrame): The dataframe to encode. Must contain only categorical data.

    Returns:
        df: The encoded dataframe.
    """
    df = df_raw.copy()
    for col in tqdm(df.columns):
        df[col] = df[col].apply(lambda x: f"{str(col)}:{str(x)}")
    return df


def bin_rare_attribute_values(df_raw, valid_cols, threshold=0.001):
    """Bins rare attribute values of a dataframe. The binning is done column-wise.
    The binning is done by replacing all attribute values that occur less than a given threshold with 'Other'.

    Args:
        df_raw (pd.DataFrame): The dataframe to bin the attribute values for.
        valid_cols (list): The list of columns to bin the attribute values for.
        threshold (float, optional): The threshold to bin the attribute values. Defaults to 0.001.

    Returns:
        df: The dataframe with binned attribute values.
    """
    # reassign 'Other' to attribute values if frequency is less than 1%
    threshold_bucket = threshold * df_raw.shape[0]
    # get all values for every column that are under the threshold
    for col in valid_cols:
        under_threshold = (
            df_raw[col]
            .value_counts()[df_raw[col].value_counts() < threshold_bucket]
            .index
        )
        df_raw[col] = df_raw[col].replace(under_threshold, "Other")
    return df_raw


def get_categories(df):
    """Returns the unique categories of a dataframe. The categories are returned column-wise.

    Args:
        df (pd.DataFrame): The dataframe to get the categories for.

    Returns:
        list: The list of list of categories.
    """
    return [list(df[c].unique()) for c in df]


def get_cardinalities(df):
    """Returns the cardinalities of a dataframe. The cardinalities are returned column-wise.

    Args:
        df (pd.DataFrame): The dataframe to get the cardinalities for.

    Returns:
        list: The list of cardinalities.
    """
    return [len(c) for c in get_categories(df)]


### FASTEST!!! ###
@st.cache_resource
def calc_frequencies_countmin_rowbased_traversal(df, max_level=None, width=False):
    """Calculates the occurrences of all combinations of attribute values in a dataframe using a CountMinSketch.
    The occurrences are calculated by traversing the unique rows of the dataframe row-wise and calculating the powerset for each row.
    Each powerset and according value count of the row are then added to the CountMinSketch.
    The width and depth of the CountMinSketch are calculated based on the expected number of keys.

    Args:
        df (pd.DataFrame): The dataframe to calculate the occurrences for.
        max_level (int, optional): The maximum length of combinations to calculate. Defaults to no. of columns.
        width (bool, optional): Whether to use a fixed width for the CountMinSketch. Defaults to False.

    Returns:
        groupby_col_combs: The CountMinSketch containing the occurrences of all combinations
        # keys: The dict of all combinations per level of pattern graph
    """
    if max_level is None:
        max_level = len(df.columns)
    base_dict = df.value_counts().to_dict()
    # calculate correct width and depth for countminsketch for fixed error rate
    if width:
        # one order of magnitude of max. expected number of keys
        w = len(base_dict) * len(list(powerset(df.columns))) / 10
        # closest number that is power of 2 to w
        w = 2 ** int(np.log2(w))
        # keep at least 2^14 = 16384 buckets (aka. 500KB of memory)
        if w < 16384:
            w = 16384
        # if w > 4194304:
        #     w = 4194304
        depth = 10
        print(
            f"Using width {w} and depth {depth}, aka. memory of {w * depth * 4 / 1024 /1024}MB"
        )
        groupby_col_combs = CountMinSketch(width=w, depth=depth)
    else:
        groupby_col_combs = CountMinSketch(size_mb=1024)
    keys_dict = {}

    # print("using row-based traversal")
    for k, v in tqdm(base_dict.items()):
        # create dict of all combinations of k as keys and v as value
        p_i = powerset_maxlength(k, max_level)
        for j in p_i:
            groupby_col_combs.increment(str(j), v)
            keys_dict.setdefault(len(j), set()).add(j)

    del base_dict, p_i
    return groupby_col_combs, keys_dict


### FAST ALTERNATIVE ###
def calc_frequencies_countmin_columnbased_groupby(df, max_level=None, width=False):
    """Calculates the occurrences of all combinations of columns in a dataframe using a CountMinSketch.
    The occurrences are calculated by grouping the dataframe by all combinations of columns and calculating the value counts for each group.
    The value counts are then added to the CountMinSketch.

    Args:
        df (pd.DataFrame): The dataframe to calculate the occurrences for.
        max_level (int, optional): The maximum level of combinations to calculate. Defaults to None.

    Returns:
        col_combs: The CountMinSketch containing the occurrences of all combinations
        keys: The set of all combinations
    """
    if max_level is None:
        max_level = len(df.columns) + 1
    if width:
        w = len(df.drop_duplicates()) * len(list(powerset(df.columns))) / 10
        w = 2 ** int(np.log2(w))
        if w < 16384:
            w = 16384
        # if w > 4194304:
        #     w = 4194304
        depth = 10
        col_combs = CountMinSketch(width=w, depth=depth)
    else:
        col_combs = CountMinSketch(size_mb=64)
    keys = {}

    for i in tqdm(range(1, max_level)):
        for comb in itertools.combinations(df.columns, i):
            # Calculate value counts for current combination and update keys
            counts = df.groupby(list(comb)).size()
            counts.index = counts.index.map(str)
            counts = counts.to_dict()
            if i == 1:
                counts = {str((c,)): v for c, v in counts.items()}
            col_combs.update(counts)
            keys.setdefault(i, set()).update(counts.keys())

            # Add combination and counts to col_combs
            for key, value in counts.items():
                col_combs.increment(str(key), value)

    del counts
    return col_combs, keys


def calculate_rowbased_exact(df, max_level=None):
    if max_level is None:
        max_level = len(df.columns)
    base_dict = df.value_counts().to_dict()
    groupby_col_combs = {}
    keys_dict = {}

    print("using row-based traversal")
    for k, v in tqdm(base_dict.items()):
        # create dict of all combinations of k as keys and v as value
        p_i = powerset_maxlength(k, max_level)
        for j in p_i:
            j_str = str(j)
            groupby_col_combs.setdefault(j_str, 0)
            groupby_col_combs[j_str] += v
            keys_dict.setdefault(len(j), set()).add(j)

    del base_dict, p_i
    return groupby_col_combs, keys_dict


@st.cache_data
def calc_attribute_comb_frequencies(columns, keys_dict, _frequency_count):
    """Calculate the frequency distribution for each attribute combination in a dataframe.

    Args:
        columns (list): The list of columns of the dataframe.
        keys_dict (dict): The dict of all combinations per level of pattern graph.
        _frequency_count (dict): The dict of all combinations and their frequency count.

    Returns:
        gen_occ: The dict of all attribute combinations and their frequency count.
    """
    gen_occ = {}
    # keys_dict = {}
    # for k in keys:
    #     keys_dict.setdefault(len(k), set()).add(k)

    for level in tqdm(range(1, len(columns) + 1)):
        keys_level = keys_dict.get(level, set())
        for k in keys_level:
            k_gen = tuple(x.split(":")[0] for x in k)
            if k_gen not in gen_occ:
                gen_occ[k_gen] = []
            gen_occ[k_gen].append(_frequency_count[str(k)])

    return gen_occ


# def get_general_occurences_catbased(categories, frequency_count):
#     """Calculate the frequency distribution for each attribute combination in a dataframe."""
#     gen_occ = {}
#     for level in tqdm(range(1, len(categories) + 1)):
#         for k in itertools.combinations(categories, level):
#             k_level = set(itertools.product(*k))
#             for k_ in k_level:
#                 k_gen = tuple(x.split(":")[0] for x in k_)
#                 if k_gen not in gen_occ:
#                     gen_occ[k_gen] = []
#                 gen_occ[k_gen].append(frequency_count[str(k_)])

#     return gen_occ


def get_frequency_count(df, max_level=None):
    """Calculates the occurrences of all combinations of attribute values in a dataframe.
    Depending on the size of the combination space and no. of unique rows,
    the occurrences can be calculated using different methods. Current default is row-based traversal.

    Args:
        df (pd.DataFrame): The dataframe to calculate the occurrences for.
        max_level (int, optional): The maximum level of combinations to calculate. Defaults to None.

    Returns:
        the occurrences of each combination of attribute-values.
    """
    print("Using row-based traversal")
    return calc_frequencies_countmin_rowbased_traversal(df, max_level)
