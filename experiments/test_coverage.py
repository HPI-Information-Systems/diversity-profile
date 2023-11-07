# %%
from datetime import date
import pandas as pd
import numpy as np
import pickle
import timeit
import time

from jpype import *

import src.coverage_countbased_exp as cov
import src.occurence_estimation_exp as occ


# %%
acs_income = pd.read_csv("data/df_ACSIncome_enc_num.csv")

# %%
bluenile = pd.read_csv("data/df_num_diamonds_enc_num.csv")

# %%
uk_roadsafety = pd.read_csv("data/df_uk_road_accident_enc_num.csv")


def get_coverage_java(threshold, df):
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

    return t, mups, valid_cols


# %%
def cov_threshold_test(test_dataset, test_name):
    repeat = 3
    cov_runtime_dict = {}
    for t_ in [1, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        if t_ == 1:
            t_ = 0
            t = 1
        else:
            t = int(t_ * test_dataset.shape[0])
            if t < 2:
                continue
        print(f"{test_name} - threshold:", t)
        # read pickle
        with open(
            f"results/{test_name}_rowbased_results_freq_output_pattern_2023-10-17.pkl",
            "rb",
        ) as f:
            rowbased_output = pickle.load(f)
        k, v = list(rowbased_output.items())[-1]
        cov_runtime_dict.setdefault(f"{test_name}_{k}", dict())
        freq_counts, keys, cols = v
        categories = occ.get_categories(test_dataset[list(cols)])
        # calculate coverage with CoverageJava (reduced cardinality)
        df = test_dataset.copy()
        cov_java_time, _, cov_java_cols = get_coverage_java(t, df)
        cov_runtime_dict[f"{test_name}_{k}"].setdefault(
            "CoverageJava", dict()
        ).setdefault(t_, [cov_java_time, cov_java_cols])
        # calculate coverage baseline
        ucov_baseline_runtime = timeit.repeat(
            lambda: cov.baseline_coverage_with_keys_all_combs(
                keys, categories, freq_counts, max_level=None, threshold=t
            ),
            repeat=repeat,
            number=1,
        )
        ucov_baseline_runtime_mean = np.mean(ucov_baseline_runtime)
        ucov_baseline_runtime_std = np.std(ucov_baseline_runtime)
        cov_runtime_dict[f"{test_name}_{k}"].setdefault(
            "Baseline (All Combs)", dict()
        ).setdefault(t_, [ucov_baseline_runtime_mean, cols])
        # calculate coverage with mups baseline
        # mups_baseline_runtime = timeit.repeat(
        #     lambda: cov.baseline_coverage_with_keys_searchbased(
        #         keys, categories, freq_counts, threshold=t
        #     ),
        #     repeat=repeat,
        #     number=1,
        # )
        # mups_baseline_runtime_mean = np.mean(mups_baseline_runtime)
        # mups_baseline_runtime_std = np.std(mups_baseline_runtime)
        # cov_runtime_dict[f"{test_name}_{k}"].setdefault(
        #     "Baseline (MUPs)", dict()
        # ).setdefault(t_, mups_baseline_runtime_mean)
        # calculate coverage with mups pwalk frequencies
        mups_pwalk_runtime = timeit.repeat(
            lambda: cov.pwalk_optimized_frequency_weight(
                categories, freq_counts, threshold=t
            ),
            repeat=repeat,
            number=1,
        )
        mups_pwalk_runtime_mean = np.mean(mups_pwalk_runtime)
        mups_pwalk_runtime_std = np.std(mups_pwalk_runtime)
        cov_runtime_dict[f"{test_name}_{k}"].setdefault(
            "MUPs pwalk", dict()
        ).setdefault(t_, [mups_pwalk_runtime_mean, cols])
    return cov_runtime_dict


# %%
def cov_combsize_test(test_dataset, test_name):
    cov_combsize_dict = {}
    repeat = 3
    t = 1
    with open(
        f"results/{test_name}_rowbased_results_freq_output_pattern_2023-10-17.pkl", "rb"
    ) as f:
        rowbased_output = pickle.load(f)
    for k, v in rowbased_output.items():
        cov_combsize_dict.setdefault(f"{test_name}", dict())
        freq_counts, keys, cols = v
        categories = occ.get_categories(test_dataset[list(cols)])
        # calculate coverage with CoverageJava (reduced cardinality)
        df = test_dataset[list(cols)].copy()
        cov_java_time, cov_java_mups, cov_java_cols = get_coverage_java(t, df)
        cov_combsize_dict[f"{test_name}"].setdefault("CoverageJava", dict()).setdefault(
            k, [cov_java_time, len(cov_java_mups), cov_java_cols]
        )
        # calculate coverage baseline
        ucov_baseline_runtime = timeit.repeat(
            lambda: cov.baseline_coverage_with_keys_all_combs(
                keys, categories, freq_counts, max_level=None, threshold=t
            ),
            repeat=repeat,
            number=1,
        )
        ucov_baseline_runtime_mean = np.mean(ucov_baseline_runtime)
        ucov_baseline_runtime_std = np.std(ucov_baseline_runtime)
        cov_combsize_dict[f"{test_name}"].setdefault(
            "Baseline (All Combs)", dict()
        ).setdefault(k, [ucov_baseline_runtime_mean, cols])
        # calculate coverage with mups baseline
        # mups_baseline_runtime = timeit.repeat(
        #     lambda: cov.baseline_coverage_with_keys_searchbased(
        #         keys, categories, freq_counts, threshold=t
        #     ),
        #     repeat=repeat,
        #     number=1,
        # )
        # mups_baseline_runtime_mean = np.mean(mups_baseline_runtime)
        # mups_baseline_runtime_std = np.std(mups_baseline_runtime)
        # cov_combsize_dict[f"{test_name}"].setdefault(
        #     "Baseline (MUPs)", dict()
        # ).setdefault(k, mups_baseline_runtime_mean)
        # calculate coverage with mups pwalk frequencies
        mups_pwalk_runtime = timeit.repeat(
            lambda: cov.pwalk_optimized_frequency_weight(
                categories, freq_counts, threshold=t
            ),
            repeat=repeat,
            number=1,
        )
        mups_pwalk_runtime_mean = np.mean(mups_pwalk_runtime)
        mups_pwalk_runtime_std = np.std(mups_pwalk_runtime)
        cov_combsize_dict[f"{test_name}"].setdefault("MUPs pwalk", dict()).setdefault(
            k, [mups_pwalk_runtime_mean, cols]
        )
    return cov_combsize_dict


# %%
def cov_combsize_test_mups(test_dataset, test_name):
    cov_combsize_dict = {}
    repeat = 1
    t = 1
    with open(
        f"results/{test_name}_rowbased_results_freq_output_pattern_2023-10-17.pkl", "rb"
    ) as f:
        rowbased_output = pickle.load(f)
    for k, v in rowbased_output.items():
        cov_combsize_dict.setdefault(f"{test_name}", dict())
        freq_counts, keys, cols = v
        categories = occ.get_categories(test_dataset[list(cols)])
        # calculate coverage with CoverageJava (reduced cardinality)
        df = test_dataset[list(cols)].copy()
        cov_time, mups, cov_java_cols = get_coverage_java(t, df)
        cov_combsize_dict[f"{test_name}"].setdefault("CoverageJava", dict()).setdefault(
            k, [len(mups), cov_java_cols, cov_time]
        )
        # calculate coverage with mups pwalk frequencies
        mups_pwalk = cov.pwalk_optimized_frequency_weight(
            categories, freq_counts, threshold=t
        )
        cov_combsize_dict[f"{test_name}"].setdefault("MUPs pwalk", dict()).setdefault(
            k, [mups_pwalk, cols]
        )
    return cov_combsize_dict


# %% [markdown]
# ### Compare Coverage Algorithms
test_datasets = [bluenile, acs_income, uk_roadsafety]
test_names = ["BlueNile", "ACSIncome", "UKRoadSafety"]

for i, test_dataset in enumerate(test_datasets):
    cov_combsize_runtime_dict = cov_combsize_test(test_dataset, test_names[i])
    with open(
        f"results/{test_names[i]}_cov_combsize_runtime_dict_mean_{date.today()}.pkl",
        "wb",
    ) as f:
        pickle.dump(cov_combsize_runtime_dict, f)
    cov_threshold_runtime_dict = cov_threshold_test(test_dataset, test_names[i])
    with open(
        f"results/{test_names[i]}_cov_threshold_runtime_dict_mean_{date.today()}.pkl",
        "wb",
    ) as f:
        pickle.dump(cov_threshold_runtime_dict, f)
    # cov_combsize_mups = cov_combsize_test_mups(test_dataset, test_names[i])
    # with open(
    #     f"results/{test_names[i]}_cov_mups_size_dict_{date.today()}.pkl", "wb"
    # ) as f:
    #     pickle.dump(cov_combsize_mups, f)
