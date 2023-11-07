## TEST FOR BEST WIDTH AND DEPTH FOR CMS STRUCTURE ##
## test with BlueNile as compatible dataset ##

## Expectation: error in frequencyes of max 0.1% for all datasets and minimal error in keys ##

import src.frequency_profiling as occ
import src.coverage as cov
import pandas as pd
import numpy as np
from bounter import CountMinSketch


def calc_rowbased_cms(base_dict, cms):
    print("using row-based traversal")
    for k, v in base_dict.items():
        # create dict of all combinations of k as keys and v as value
        p_i = occ.powerset(k)
        for j in p_i:
            cms.increment(str(j), v)
    return cms


def main(df, cols):
    base_dict = df.value_counts().to_dict()
    freq_count_cols, keys_cols = occ.calculate_rowbased_exact(df[list(cols)])
    keys = []
    for _, v in keys_cols.items():
        keys.extend(v)
    categories = occ.get_categories(df[list(cols)])
    len_actual_mups = 8375

    # compare different widths and depths for CMS with results from exact calculation
    # Test 1: compare error in collisions of keys (to set correct depth), fixed width of 128MB in bytes
    w = len(base_dict) * len(list(occ.powerset(df.columns))) / 10
    # closest number that is power of 2 to w
    w = 2 ** int(np.log2(w))
    cov_error = []
    depths = [4, 6, 8, 10, 12, 14]
    for d in depths:
        cms = CountMinSketch(width=w, depth=d)
        cms_freq_count_cols = calc_rowbased_cms(base_dict, cms)

        # calculate coverage and compare to actual amount of mups
        mups = cov.freqwalk_frequency_weight(
            categories, cms_freq_count_cols, threshold=1
        )
        len_mups = sum([len(mups[m]) for m in mups.keys()])
        cov_error_d = abs(len_actual_mups - len_mups) / len_actual_mups
        cov_error.append(round(cov_error_d, 3))
        print("width: ", w, " depth: ", d, "error: ", cov_error_d)
    # print smallest error
    print(
        "smallest error: ",
        min(cov_error),
        " with depth: ",
        depths[cov_error.index(min(cov_error))],
    )

    # Test 2: compare error in frequencies of keys (to set correct width), fixed depth of 12
    error = []
    depth = 10
    widths = [2**14, 2**16, 2**18, 2**20, 2**22, 2**24]
    ## w has to be power of 2, check for different power of 2 starting with 2**14
    for w in widths:
        cms = CountMinSketch(width=w, depth=depth)
        cms_freq_count_cols = calc_rowbased_cms(base_dict, cms)

        # calculate error in frequency for keys between exact and cms
        error_freq = []
        for key in keys:
            error_freq.append(
                cms_freq_count_cols[str(key)] - freq_count_cols[str(key)]
            )  # /freq_count_cols[str(key)]
        error_freq_mean = np.median(error_freq)
        error.append(error_freq_mean)
        print("width: ", w, "error: ", error_freq_mean)
    # print smallest median error
    print(
        "smallest median error: ",
        min(error),
        " with width: ",
        widths[error.index(min(error))],
    )


# init
df = pd.read_csv("data/temp_input_num.csv")
cols = df.columns
main(df, cols)

# result: depth of 10 has sufficiently small error for own calculation,
# could even get no error on frequencies with depth of 10 and smaller width
