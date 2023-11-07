import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import cast, Any
import plotly.express as px
import plotly.graph_objects as go

import itertools
import random
import os

import src.entropy as entropy
import src.coverage as cov
import src.frequency_profiling as occ

import src.st_output as st_out

#### Set up page config ####
st.set_page_config(
    page_title="Diversity Profile",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Diversity Profile")
# print(st.session_state)


def set_session_state():
    if (
        not "submitted" in st.session_state
        and not "submitted_var" in st.session_state
        and not "keys" in st.session_state
        and not "keys_ref" in st.session_state
        and not "len_keys_ref" in st.session_state
        and not "cardinalities_ref" in st.session_state
        and not "dataset" in st.session_state
        and not "input_data_select" in st.session_state
        and not "valid_cols" in st.session_state
        and not "frequency_general" in st.session_state
        and not "dist_topk" in st.session_state
    ):
        st.session_state["submitted"] = False
        st.session_state["submitted_var"] = None
        st.session_state["keys"] = dict()
        st.session_state["keys_ref"] = dict()
        st.session_state["len_keys_ref"] = 0
        st.session_state["cardinalities_ref"] = []
        st.session_state["frequency_general"] = dict()
        st.session_state["dataset"] = ""
        st.session_state["input_data_select"] = False
        st.session_state["valid_cols"] = []
        st.session_state["dist_topk"] = []


set_session_state()

#### Set up sidebar ####
with st.sidebar:
    st.header("Filters")

    input_data = st.file_uploader(
        "Upload a CSV file containing categorical data:",
        type=["csv"],
        key="input_data",
    )

    # check if data is uploaded, if not use default dataset
    if input_data is not None:
        input_data_select = True
        st.session_state["input_data_select"] = input_data_select
        dataset = input_data.name
        df = pd.read_csv(input_data)
        reference_dataset = "-NONE-"
        df_ref = None
    else:
        input_data_select = False
        dataset = st.selectbox(
            "OR: Choose a preloaded dataset:", st_out.FILE_NAMES.keys()
        )

        reference_dataset = st.selectbox(
            "Choose a reference dataset:",
            st_out.REFERENCE_FILE_NAMES.keys(),
            key="reference_dataset",
        )

        try:
            df, df_ref = st_out.load_data(
                st_out.FILE_NAMES[dataset],
                st_out.REFERENCE_FILE_NAMES[reference_dataset],
            )
        except Exception:
            st.error("Error loading dataset. Please try again.")
            st.stop()
    # check if data is encoded, if not encode
    if not st_out.check_file_encoded(df):
        df = occ.encode_data(df)
    if df_ref is not None and not st_out.check_file_encoded(df_ref):
        df_ref = occ.encode_data(df_ref)

    cols, cols_ref = st_out.get_column_names(df, df_ref)

    valid_cols = st.multiselect(
        "Choose attributes of interest. Selected columns will be treated as categorical variables:",
        cols,
        max_selections=6,
    )

    df = df[valid_cols]

    if cov.combinatorial_sum(occ.get_cardinalities(df)) > 500000:
        st.warning(
            "The selection contains a large amount of attribute value combinations. This may take a while to calculate."
        )

    if df_ref is not None:
        if valid_cols and not set(valid_cols).issubset(set(df_ref.columns)):
            st.error(
                "The selected reference dataset does not contain the same attributes as the selected dataset. Please choose another reference dataset."
            )
            st.stop()
        df_ref = df_ref[valid_cols]

    # add selection box to choose which attribute values shall be ignored since they are non meaningful for diversity
    # e.g. ID columns
    ignore_attribute_values = st.multiselect(
        "Ignore rows with the following attribute values:",
        sum(occ.get_categories(df), []),
        # default=cols[:2],
        key="ignore_attribute_values",
    )

    # create a slider to select maximum level to check for coverage in the dataset
    max_length = len(valid_cols) if valid_cols and len(valid_cols) > 1 else len(cols)

    max_level = st.slider(
        "Max pattern length for coverage analysis",
        min_value=1,
        max_value=max_length,
        value=max_length,
        step=1,
        key="max_level",
    )

    max_value = int(0.01 * df.shape[0]) if df is not None and df.shape[0] > 200 else 2

    # create a slider to select coverage threshold
    coverage_threshold = st.slider(
        "Coverage threshold $t$",
        min_value=1,
        max_value=max_value,
        value=1,
        step=1,
        key="coverage_threshold",
    )

    mups_only = st.checkbox(
        "Show only MUPs (faster for large datasets)", False, key="mups_only"
    )

    submitted_var = (
        input_data_select,
        dataset,
        reference_dataset,
        valid_cols,
        ignore_attribute_values,
        max_level,
        coverage_threshold,
        mups_only,
    )

    submitted = st.button("Submit")
    if submitted:
        st.session_state["submitted"] = True
        if st.session_state["submitted_var"] is None:
            st.session_state["submitted_var"] = submitted_var


#### Set up main page output ####
def main(df):
    if st.session_state["submitted_var"] == submitted_var:
        print("submitted_var did not change, keep session state...")
        st.session_state["submitted"] = True

    if st.session_state["submitted"]:
        # Caveat and Description
        st.write(
            "This app calculates the Diversity Profile of a dataset. We consider two main aspects:\n **Coverage** and **Heterogeneity**.\
               A more comprehensive and even distribution indicates a more diverse dataset.\
                To handle large datasets, we use approximations, so the results might not be perfectly precise in some cases."
        )
        # Filter dataset
        df = df[~df.isin(ignore_attribute_values).any(axis=1)]
        st.session_state.categories = occ.get_categories(df)
        st.session_state.cardinalities = occ.get_cardinalities(df)
        if df_ref is not None:
            print("initialize reference dataset...")
            init_df_ref(df_ref)

        # Calculate frequency count of all patterns
        print("calculating frequency count...")
        if (
            st.session_state["submitted_var"] != submitted_var
            or "frequency_count" not in st.session_state
        ):
            (
                st.session_state.frequency_count,
                keys,
            ) = occ.calc_frequencies_countmin_rowbased_traversal(df, max_level)
            frequencies_general = occ.calc_attribute_comb_frequencies(
                valid_cols, keys, st.session_state.frequency_count
            )
            st.session_state["keys"] = cast(Any, keys)
            st.session_state["frequency_general"] = cast(Any, frequencies_general)
        else:
            print("using cached frequency count...")
            keys = st.session_state["keys"]
            frequencies_general = st.session_state["frequency_general"]
        st.session_state.len_keys = sum(len(v) for v in keys.values())
        # if not coverage_threshold > 1:
        #     st.session_state.len_keys_covered = st.session_state.len_keys

        coverage(keys)
        st.divider()

        if df_ref is not None:
            heterogeneity_ref(frequencies_general)
        else:
            heterogeneity(frequencies_general)

        st.session_state["submitted_var"] = submitted_var
        st.session_state["submitted"] = False


def coverage(keys):
    print("presenting coverage...")
    st.header("Coverage", divider="gray")
    st.write(
        "Coverage describes the expectation that every combination of attribute-values should exist at least $t$ times in the dataset for it to be fully diverse.\
            Here we present the uncovered combinations in the dataset.\
                The coverage threshold $t$ can be adjusted in the sidebar."
    )
    col1, col2 = st.columns([0.6, 0.4], gap="medium")

    ## Calculate uncovered combinations in the dataset
    if (
        st.session_state["submitted_var"] != submitted_var
        or "mups" not in st.session_state
    ):
        if mups_only:
            print(f"calculate uncovered mups with threshold {coverage_threshold} ...")
            mups = cov.freqwalk_frequency_weight(
                st.session_state.categories,
                st.session_state.frequency_count,
                threshold=st.session_state.coverage_threshold,
            )
            # st.session_state.len_keys = sum(len(v) for v in keys.values())
        else:
            print(f"calculate uncovered combs with threshold {coverage_threshold}...")
            mups = cov.baseline_coverage_with_keys_all_combs(
                keys,
                st.session_state.categories,
                st.session_state.frequency_count,
                max_level,
                coverage_threshold,
            )
            st.session_state.len_keys = sum(len(v) for v in keys.values()) - sum(
                len(v) for v in mups.values()
            )

        st.session_state.mups = cast(Any, mups)
    else:
        print("using cached uncovered combinations...")
        mups = st.session_state.mups

    st.session_state.len_mups = sum(len(v) for v in mups.values())

    with col1:
        coverage_statistics(mups)
    with col2:
        coverage_patterns(mups)


def heterogeneity(frequencies_general):
    print("calculating heterogeneity...")
    st.header("Heterogeneity", divider="gray")
    st.write(
        "Heterogeneity describes the variety and distribution of attribute-values.\
             Here we present the most uniform and non-uniform attribute combinations."
    )
    col3, col4 = st.columns([0.6, 0.4], gap="medium")

    with col4:
        dist_to_uniform = entropy.get_dist_from_expected_distribution(
            frequencies_general
        )
        heterogeneity_patterns(dist_to_uniform)

    with col3:
        heterogeneity_statistics(dist_to_uniform, ref=False)
    entropy_vis(dist_to_uniform, ref=False)


def heterogeneity_ref(frequencies_general):
    print("calculating heterogeneity...")
    st.header("Heterogeneity", divider="gray")
    col3, col4 = st.columns([0.6, 0.4], gap="medium")

    with col4:
        dist_to_uniform = entropy.get_dist_from_expected_distribution(
            frequencies_general
        )
        frequencies_general_ref = occ.calc_attribute_comb_frequencies(
            valid_cols, st.session_state.keys_ref, st.session_state.frequency_count_ref
        )
        dist_to_uniform_ref = entropy.get_dist_from_expected_distribution(
            frequencies_general_ref
        )
        heterogeneity_patterns_ref(dist_to_uniform, dist_to_uniform_ref)

    with col3:
        heterogeneity_statistics(dist_to_uniform, ref=True)
    entropy_vis(dist_to_uniform, ref=True)


def init_df_ref(df_ref):
    print("initializing reference dataset...")
    st.session_state.cardinalities_ref = occ.get_cardinalities(df_ref)
    if (
        st.session_state["submitted_var"] != submitted_var
        or "frequency_count_ref" not in st.session_state
    ):
        (
            st.session_state.frequency_count_ref,
            keys_ref,
        ) = occ.calc_frequencies_countmin_rowbased_traversal(df_ref, max_level)
        st.session_state["keys_ref"] = cast(Any, keys_ref)
    else:
        print("using cached frequency count...")
        keys_ref = st.session_state["keys_ref"]
    st.session_state.len_keys_ref = sum(len(v) for v in keys_ref.values())
    print("Ref KEYS", st.session_state.len_keys_ref)


def coverage_statistics(mups):
    st.subheader("""Coverage Statistics""")

    len_keys = st.session_state.len_keys
    len_keys_ref = st.session_state.len_keys_ref
    len_mups = st.session_state.len_mups
    cardinalities = st.session_state.cardinalities

    st.write("Cardinalities:\t")

    if df_ref is not None:
        cardinalities_ref = st.session_state.cardinalities_ref
        st.table(
            pd.DataFrame(
                [cardinalities, cardinalities_ref],
                columns=df_ref.columns,
                index=["Cardinality (Dataset)", "Cardinality (Reference)"],
            )
        )
        st.write(
            f"Amount of covered attribute-value combinations:\t{str(len_keys)}\t ({str(round((len_keys/len_keys_ref)*100,2))}% of all available combinations in reference set)"
        )
    else:
        st.table(
            pd.DataFrame(cardinalities, index=df.columns, columns=["Cardinality"]).T
        )
        st.write(
            f"Amount of attribute-value combinations:\t{str(len_keys)}\t ({str(round((len_keys/cov.combinatorial_sum(cardinalities))*100,2))}% of all possible combinations)"
        )
    if mups_only:
        st.write("No. of Maximal Uncovered Patterns (MUPs):\t" + str(len_mups))
    else:
        st.write("Uncovered Combinations:\t" + str(len_mups))

    st.write("Rare Attributes (<1%): ")
    level1_rare_attributes = cov.get_rare_attributes(
        st.session_state.categories, st.session_state.frequency_count, df.shape[0]
    )
    rare_attributes_df = pd.DataFrame.from_dict(
        level1_rare_attributes,
        columns=["Count", "Percentage"],
        orient="index",
    )
    rare_attributes_df["Percentage"] = rare_attributes_df["Percentage"].apply(
        lambda x: str(round(x, 2)) + " %"
    )

    if df_ref is not None:
        rare_attributes_df.columns = ["Count (Dataset)", "Percentage (Dataset)"]
        rare_attributes_df["Count (Reference)"] = [
            st.session_state.frequency_count_ref[str((i,))]
            for i in level1_rare_attributes.keys()
        ]
        rare_attributes_df["Percentage (Reference)"] = [
            str(
                round(
                    (st.session_state.frequency_count_ref[str((i,))] / df_ref.shape[0])
                    * 100,
                    3,
                )
            )
            + " %"
            if st.session_state.frequency_count_ref[str((i,))] != 0
            else None
            for i in level1_rare_attributes.keys()
        ]

    # st.dataframe of rare_attributes_df so that there is a bold line between column Percentage(Dataset) and Count(Reference
    st.dataframe(
        rare_attributes_df,
        height=None,
        use_container_width=True,
        column_config={"Percentage (Dataset)": {"bold": True}},
    )

    if mups_only:
        mups_per_level = pd.DataFrame.from_dict(
            {k: len(v) for k, v in mups.items()},
            orient="index",
            columns=["MUPs per Level"],
        )
        st.write("Distribution of MUPs per level:")
    else:
        mups_per_level = pd.DataFrame.from_dict(
            {k: len(v) for k, v in mups.items()},
            orient="index",
            columns=["Uncovered Combinations per Level"],
        )
        st.write("Distribution of uncovered combinations per level:")
    st.bar_chart(
        mups_per_level,
        height=300,
        use_container_width=True,
    )


def coverage_patterns(mups):
    st.subheader("""Coverage Pattern Occurrences""")

    TOPK_MUPS = 5 if st.session_state.len_mups > 5 else st.session_state.len_mups

    # choose level of interest, default to level 2
    cov_level = st.selectbox(
        "Select level of interest:",
        list(range(1, max_level + 1)),
        index=0 if 1 in mups.keys() else 1,
        key="cov_level",
    )

    # get mups of selected level with lowest total occurence percentage of higher level parents
    mups_level = mups.get(cov_level, set())
    print(f"mups for cov_level {int(cov_level)}: ", len(mups_level))
    mups_topk = cov.get_topk_mups(
        mups_level, st.session_state.frequency_count, TOPK_MUPS
    )
    st.write(
        "**Top "
        + str(TOPK_MUPS)
        + " uncovered patterns (ranked by total frequency of parent patterns):**"
    )
    for mup in mups_topk:
        st.write(mup)
        with st.expander("**Occurrences of parent patterns:**"):
            for (
                combination,
                occurence,
            ) in cov.get_occurences_of_parent_patterns(
                mup, st.session_state.frequency_count
            ).items():
                st.text(
                    f"{', '.join(combination)} : {str(round((occurence/ df.shape[0]) * 100, 3))}%"
                )
    st.divider()
    with st.expander(f"All other uncovered patterns in Level {cov_level}"):
        for mup in mups_level:
            if mup not in mups_topk:
                st.write(mup)


def heterogeneity_patterns(dist_to_uniform):
    st.subheader("""Heterogeneity Patterns""")
    TOPK = 5
    st.write(
        f"Top {TOPK} most non-uniform attribute combinations (between 0 (uniform) and 1):"
    )
    dist_topk = list(dist_to_uniform.keys())[:TOPK]
    st.session_state.dist_topk = dist_topk
    dist_topk_values = list(dist_to_uniform.values())[:TOPK]
    # filter dist_to_uniform for keys in dist_topk and display as st.table
    st.dataframe(
        pd.DataFrame(
            {
                "Pattern": [", ".join(k) for k in dist_topk],
                "Distance to uniform distribution": [
                    str(np.round(v, 3)) for v in dist_topk_values
                ],
            },
        ),
        hide_index=True,
    )
    st.write(
        f"Top {TOPK} most uniform attribute combinations (between 0 (uniform) and 1):"
    )
    dist_lowk = list(dist_to_uniform.keys())[-TOPK:][::-1]
    dist_lowk_values = list(dist_to_uniform.values())[-TOPK:][::-1]
    # filter dist_to_uniform for keys in dist_lowk and display as st.table
    st.dataframe(
        pd.DataFrame(
            {
                "Pattern": [", ".join(k) for k in dist_lowk],
                "Distance to uniform distribution": [
                    str(np.round(v, 3)) for v in dist_lowk_values
                ],
            }
        ),
        hide_index=True,
    )


def heterogeneity_patterns_ref(dist_to_uniform, dist_to_uniform_ref):
    st.subheader("""Heterogeneity Patterns""")
    TOPK = 5

    st.write("Top " + str(TOPK) + " most non-uniform patterns:")
    dist_topk = list(dist_to_uniform.keys())[:TOPK]
    st.session_state.dist_topk = dist_topk
    dist_topk_values = list(dist_to_uniform.values())[:TOPK]
    dist_topk_values_ref = list(dist_to_uniform_ref.values())[:TOPK]
    # filter dist_to_uniform for keys in dist_topk and display as st.table
    st.table(
        pd.DataFrame(
            {
                "Pattern": [", ".join(k) for k in dist_topk],
                "Distance to uniform distribution": [
                    str(np.round(v, 3)) for v in dist_topk_values
                ],
                "Distance to uniform distribution (Reference)": [
                    str(np.round(v, 3)) for v in dist_topk_values_ref
                ],
            }
        )
    )

    st.write("Top " + str(TOPK) + " most uniform attribute combination:")
    dist_lowk = list(dist_to_uniform.keys())[-TOPK:][::-1]
    dist_lowk_values = list(dist_to_uniform.values())[-TOPK:][::-1]
    dist_lowk_values_ref = list(dist_to_uniform_ref.values())[-TOPK:][::-1]
    # filter dist_to_uniform for keys in dist_lowk and display as st.table
    st.table(
        pd.DataFrame(
            {
                "Pattern": [", ".join(k) for k in dist_lowk],
                "Distance to uniform distribution": [
                    str(np.round(v, 3)) for v in dist_lowk_values
                ],
                "Distance to uniform distribution (Reference)": [
                    str(np.round(v, 3)) for v in dist_lowk_values_ref
                ],
            }
        )
    )


def heterogeneity_statistics(dist_to_uniform, ref=False):
    st.subheader("""Heterogeneity Statistics""")
    if df_ref is not None:
        st.write(
            "Total number of rows:\t"
            + "{:,}".format(df.shape[0])
            + f"\t ({'{:,}'.format(df_ref.shape[0])} in reference dataset)"
        )
    else:
        st.write(
            "Total number of rows:\t"
            + "{:,}".format(df.shape[0])
            + "\t**|**\t"
            + "Total number of columns:\t"
            + "{:,}".format(df.shape[1])
        )
    st.subheader("""Distribution of Attributes""")
    dist_column_default = list(st.session_state.dist_topk[0])
    dist_column_default.sort(key=lambda x: dist_to_uniform[(x,)], reverse=True)
    dist_columns = st.multiselect(
        "Select column(s) of interest [defaults to the most non-uniform column combination and sorted by uniformity of attribute]:",
        list(df.columns),
        default=dist_column_default,
    )
    if not dist_columns:
        st.warning("Please select at least one column.")
    else:
        if ref:
            st_out.show_distribution_with_ref(df, df_ref, dist_columns)
        else:
            st_out.show_distribution(df, dist_columns)


def entropy_vis(dist_to_uniform, ref=False):
    print("presenting entropy...")
    st.subheader("""Entropy""")

    entropy_list = list(itertools.permutations(valid_cols, 2))
    entropy_list.sort(
        key=lambda x: dist_to_uniform[x]
        if x in dist_to_uniform.keys()
        else dist_to_uniform[(x[1], x[0])],
        reverse=True,
    )

    entropy_cols = st.selectbox(
        "Select attribute combination to calculate Shannon Diversity Index for:",
        entropy_list,
        index=0,
    )

    st.write(
        f"**Shannon Diversity Index of {entropy_cols[0]} regarding {entropy_cols[1]}:**\n\nError bars display the deviation from maximum achievable diversity in each Diversity Index when assuming a uniform distribution for every available combination of {entropy_cols[1]} and {entropy_cols[0]}. Note that not in every {entropy_cols[0]} group all possible combinations are available."
    )

    (
        true_divs,
        shannon_divs,
        max_even_shannon_divs,
    ) = entropy.get_categorical_true_diversity(
        df,
        st.session_state.frequency_count,
        entropy_cols[0],
        entropy_cols[1],
        st.session_state.categories,
    )
    if not ref:
        st_out.show_true_diversity(
            shannon_divs,
            max_even_shannon_divs,
            entropy_cols,
            y_length=st.session_state.cardinalities[
                df.columns.get_loc(entropy_cols[1])
            ],
        )
        hill_label = "Hill Numbers"
    else:
        (
            true_divs_ref,
            shannon_divs_ref,
            max_even_shannon_divs_ref,
        ) = entropy.get_categorical_true_diversity(
            df_ref,
            st.session_state.frequency_count_ref,
            entropy_cols[0],
            entropy_cols[1],
            occ.get_categories(df_ref),
        )
        st_out.show_true_diversity_with_ref(
            shannon_divs,
            max_even_shannon_divs,
            shannon_divs_ref,
            max_even_shannon_divs_ref,
            entropy_cols,
        )
        hill_label = "Hill Numbers (Dataset)"

    with st.expander(hill_label):
        # frequency distribution of entropy_cols
        frequency_general_entropy_cols = {}
        for k, v in st.session_state.frequency_general.items():
            if (
                any((entropy_cols[i],) == k for i in range(len(entropy_cols)))
                or entropy_cols == k
            ):
                frequency_general_entropy_cols[k] = list(v)
        st_out.stplot_hill_numbers(frequency_general_entropy_cols)

        max_true_divs_0 = int(max(list(true_divs[i][0] for i in true_divs.keys())))
        max_true_divs_3 = int(max(list(true_divs[i][10] for i in true_divs.keys())))
        if (max_true_divs_0 - max_true_divs_3) > 1:
            st.write(
                f"In the following scenario, the data on the distribution of {entropy_cols[1]} in a certain {entropy_cols[0]} group may show that there are max. {max_true_divs_0} different groups of {entropy_cols[1]} represented within the population (q=0), but there is a reduced effective diversity due to the dominance of one or a few groups in {entropy_cols[1]} when accounting for their proportional representation."
            )
        else:
            st.write(
                f"In the following scenario, the data on the distribution of {entropy_cols[1]} in a certain {entropy_cols[0]} group may show that there are max. {max_true_divs_0} different groups of {entropy_cols[1]} represented within the population (q=0) and there is only little dominance of one or a few groups in {entropy_cols[1]} when accounting for their proportional representation."
            )

        true_divs["uniform dist_" + ", ".join(entropy_cols)] = [
            list(true_divs.values())[0][0]
        ] * len(list(np.arange(0, 3.1, 0.1)))
        st.plotly_chart(
            px.line(
                pd.DataFrame(true_divs, index=list(np.arange(0, 3.1, 0.1))),
                labels={
                    "index": "q",
                    "value": "True Diversity",
                },
            ),
            theme="streamlit",
            use_container_width=True,
        )


main(df)
