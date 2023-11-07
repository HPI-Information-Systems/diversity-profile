# %%
import pandas as pd

import itertools
import sys
import pickle

import src.occurence_estimation_exp as occ

from time import time
from datetime import date
from tqdm import tqdm

# %% [markdown]
# ### Perform Runtime Tests


# %%
def combinatorial_sum(numbers):
    """Calculates the sum of all possible combinations of a list of cardinalities.
    Example:
    >>> combinatorial_sum([1, 2, 3])
    20

    Args:
        numbers (list): list of cardinalities

    Returns:
        int: sum of all possible combinations of the cardinalities
    """
    result = 0
    n = len(numbers)

    for r in range(1, n + 1):
        for combo in itertools.combinations(numbers, r):
            product = 1
            for num in combo:
                product *= num
            result += product

    return result


# %%
def get_itemdict_of_combinatorial_sum(df):
    unique_values = {col: df[col].unique() for col in df.columns}
    cardinality = [f"{col}#{len(unique_values[col])}" for col in unique_values.keys()]
    card_ps = itertools.chain.from_iterable(
        itertools.combinations(cardinality, r) for r in range(2, len(cardinality) + 1)
    )

    cardinality_combinations = {
        tuple(col.split("#")[0] for col in i): combinatorial_sum(
            [int(col.split("#")[1]) for col in i]
        )
        for i in tqdm(card_ps)
    }

    # Sorting the combinations by their combinatorial sum values and combination length
    cardinality_combinations = dict(
        sorted(cardinality_combinations.items(), key=lambda x: (x[1], len(x[0])))
    )

    # for every combination length, take the combination with the largest combinatorial sum
    selected_combinations = {}
    for k, v in cardinality_combinations.items():
        if len(k) not in selected_combinations.keys():
            selected_combinations[len(k)] = [k, v]
        else:
            if v > selected_combinations[len(k)][1]:
                selected_combinations[len(k)] = [k, v]

    return selected_combinations


# %%
def rowbased_runtime_performance(name, df, buckets, max_level=None):
    runtime_dict = {}
    output_dict = {}
    for cols, combination_size in tqdm(buckets.values()):
        t = time()
        freq_count_rows, keys_rows = occ.calc_occurences_countmin_rowbased_traversal(
            df[list(cols)], max_level=max_level, width=True
        )
        runtime = time() - t
        memory_size_keys = sum([sys.getsizeof(v) for v in keys_rows.values()])
        memory_size_cms = freq_count_rows.size()
        no_of_keys = sum(len(v) for v in keys_rows.values())
        runtime_dict[no_of_keys] = [
            runtime,
            combination_size,
            memory_size_keys,
            memory_size_cms,
            freq_count_rows.quality(),
            len(cols),
            cols,
        ]
        output_dict[no_of_keys] = [freq_count_rows, keys_rows, cols]
    with open(
        f"results/{name}_rowbased_results_freq_output_pattern_{date.today()}.pkl", "wb"
    ) as f:
        pickle.dump(output_dict, f)
    with open(
        f"results/{name}_rowbased_runtime_freq_output_pattern_{date.today()}.pkl", "wb"
    ) as f:
        pickle.dump(runtime_dict, f)


# %%
def columnbased_runtime_performance(name, df, buckets, max_level=None):
    runtime_dict = {}
    output_dict = {}
    for cols, combination_size in tqdm(buckets.values()):
        t = time()
        freq_count_cols, keys_cols = occ.calc_occurences_countmin_columnbased_groupby(
            df[list(cols)], max_level=max_level, width=True
        )
        runtime = time() - t
        memory_size_keys = sum([sys.getsizeof(v) for v in keys_cols.values()])
        memory_size_cms = freq_count_cols.size()
        no_of_keys = sum(len(v) for v in keys_cols.values())
        runtime_dict[no_of_keys] = [
            runtime,
            combination_size,
            memory_size_keys,
            memory_size_cms,
            freq_count_cols.quality(),
            len(cols),
            cols,
        ]
        output_dict[no_of_keys] = [freq_count_cols, keys_cols, cols]
    with open(
        f"results/{name}_columnbased_results_freq_output_pattern_{date.today()}.pkl",
        "wb",
    ) as f:
        pickle.dump(output_dict, f)
    with open(
        f"results/{name}_columnbased_runtime_freq_output_pattern_{date.today()}.pkl",
        "wb",
    ) as f:
        pickle.dump(runtime_dict, f)


# %%
def exact_runtime_performance(name, df, buckets, max_level=None):
    runtime_dict = {}
    output_dict = {}
    for cols, combination_size in tqdm(buckets.items()):
        t = time()
        freq_count_cols, keys_cols = occ.calculate_rowbased_exact(
            df[list(cols)], max_level=max_level
        )
        runtime = time() - t
        memory_size_keys = sum([sys.getsizeof(v) for v in keys_cols.values()])
        memory_size_dict = sys.getsizeof(freq_count_cols)
        no_of_keys = sum(len(v) for v in keys_cols.values())
        runtime_dict[no_of_keys] = [
            runtime,
            combination_size,
            memory_size_keys,
            memory_size_dict,
            None,
            cols,
        ]
        output_dict[no_of_keys] = [freq_count_cols, keys_cols, cols]
    with open(
        f"results/{name}_exact_results_freq_output_pattern_{date.today()}.pkl", "wb"
    ) as f:
        pickle.dump(output_dict, f)
    with open(
        f"results/{name}_exact_runtime_freq_output_pattern_{date.today()}.pkl", "wb"
    ) as f:
        pickle.dump(runtime_dict, f)


# %%
def calculate_runtime_values(name, df, buckets, max_level):
    print(f"Calculating runtime values for {name}...")
    # Group-By Rowbased CountMin Sketch [Proposed Approach]
    # Dependent on no. of unique rows in the dataset (base patterns)
    print("Calculating rowbased runtime values...")
    rowbased_runtime_performance(name, df, buckets, max_level=max_level)
    # Group-By Columnbased CountMin Sketch [Baseline Approach]
    # Dependent on no. of unique combinations in the dataset
    print("Calculating columnbased runtime values...")
    columnbased_runtime_performance(name, df, buckets, max_level=max_level)
    # Exact Counting [Baseline Approach]
    # Dependent on no. of unique combinations in the dataset
    # print("Calculating exact runtime values...")
    # exact_runtime_performance(name, df, buckets, max_level=max_level)


# %% [markdown]
# ### Load Test Datasets
# - ACSIncome (data/df_ACSIncome_enc.csv)
# - UK Road Safety (data/df_uk_road_accident_enc.csv)
# - BlueNile (data/df_diamonds_enc.csv)

# %%
acs_income = pd.read_csv("data/df_ACSIncome_enc_num.csv")

# %%
blue_nile = pd.read_csv("data/df_diamonds_enc_num.csv")

# %%
uk_roadsafety = pd.read_csv("data/df_uk_road_accident_enc_num.csv")

# %% [markdown]
# #### Frequency Count

# %% [markdown]
# ##### BlueNile

# # %%
bluenile_buckets = get_itemdict_of_combinatorial_sum(blue_nile)
calculate_runtime_values("BlueNile", blue_nile, bluenile_buckets, max_level=None)

# %% [markdown]
# ##### ACSIncome

# %%
acs_income_buckets = get_itemdict_of_combinatorial_sum(acs_income)
calculate_runtime_values("ACSIncome", acs_income, acs_income_buckets, max_level=None)

# %% [markdown]
# ##### UK Road Safety Data

# %%
uk_roadsafety_buckets = get_itemdict_of_combinatorial_sum(uk_roadsafety)
calculate_runtime_values(
    "UKRoadSafety", uk_roadsafety, uk_roadsafety_buckets, max_level=None
)
