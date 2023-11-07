import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import itertools
import random
import os

import src.entropy as entropy
import src.coverage as cov
import src.frequency_profiling as occ

FILE_NAMES = {
    "Toy Example": "data/toy_example.csv",
    "COMPAS": "data/df_compas.csv",
    "BlueNile": "data/df_diamonds.csv",
    #"Toy Example 2": "data/toy_example2.csv",
    #"Adult": "data/df_uci_adult.csv",
    # "Adult_2018": "data/df_ACSIncome_enc.csv",
    #"Adult_2018_CA": "data/df_ACSIncome_CA_enc.csv",
    #"UK RoadSafety Accident": "data/df_uk_road_accident_enc.csv",
    #"IMDb_Top": "data/imdb.csv",
    #"Covertype": "data/covertype.csv",
}

REFERENCE_FILE_NAMES = {
    "-NONE-": None,
    #"COMPAS_random": "data/df_compas_ideal.csv",
    #"Adult_reconstructed": "data/df_adult.csv",
    #"Adult_biased": "data/df_uci_adult_mw.csv",
    #"Adult_2018": "data/df_ACSIncome_enc.csv",
}


@st.cache_data
def load_data(df_filepath, reference_filepath=None):
    df = pd.read_csv(df_filepath)
    if reference_filepath is None:
        return df, None
    df_ref = pd.read_csv(reference_filepath)
    return df, df_ref


def check_file_encoded(df):
    """Checks if the dataset is encoded by checking if every element starts with its
    corresponding attribute name followed by ":".

    Args:
        df (pd.DataFrame): The dataset to check.

    Returns:
        bool: True if the dataset is encoded, False otherwise.
    """
    # check if every element starts with its corresponding attribute name followed by ":"
    return all(
        [str(df.iloc[0, i]).startswith(df.columns[i] + ":") for i in range(df.shape[1])]
    )


def get_column_names(df, df_ref=None):
    if df_ref is not None:
        return df.columns.to_list(), df_ref.columns.to_list()
    else:
        return df.columns.to_list(), None


@st.cache_resource
def show_distribution(df, dist_columns):
    """Displays a plotly treemap with the distribution of the selected columns in
    the dataset.

    Args:
        df (pd.DataFrame): The dataset to display the distribution for.
        dist_columns (list): The columns to display the distribution for.
    """
    # calculate frequency count of all patterns filtered for selected columns (faster than using CountMinSketch)
    dist_chart_data = (
        df.groupby(dist_columns).size().sort_values(ascending=False).reset_index()
    )
    dist_chart_data.columns = dist_columns + ["Count"]

    # display attributes as plotly treemap
    st.plotly_chart(
        px.treemap(
            dist_chart_data,
            path=dist_columns,
            values="Count",
        ),
        use_container_width=True,
    )


@st.cache_resource
def show_distribution_with_ref(df, df_ref, dist_columns):
    """Displays a plotly bar chart with the distribution of the selected columns in
    the dataset and the reference dataset.

    Args:
        df (pd.DataFrame): The dataset to display the distribution for.
        df_ref (pd.DataFrame): The reference dataset to display the distribution for.
        dist_columns (list): The columns to display the distribution for.

    """
    dist_chart_data = df.groupby(dist_columns).size().sort_values(ascending=False)
    # convert data into percentages
    dist_chart_data = round(dist_chart_data / df.shape[0], 3) * 100
    dist_chart_data_ref = (
        df_ref.groupby(dist_columns).size().sort_values(ascending=False)
    )
    # convert data into percentages
    dist_chart_data_ref = round(dist_chart_data_ref / df_ref.shape[0], 3) * 100

    # only join index if multiindex
    if isinstance(dist_chart_data.index, pd.MultiIndex):
        dist_chart_data.index = dist_chart_data.index.map(" & ".join)
        dist_chart_data_ref.index = dist_chart_data_ref.index.map(" & ".join)

    # sort dist_chart_data_ref descending
    dist_chart_data_ref = dist_chart_data_ref.sort_values(ascending=False)

    # sort dist_chart_data index by dist_chart_data_ref values if possible
    # if dist_chart_data.index.equals(dist_chart_data_ref.index):
    dist_chart_data = dist_chart_data.reindex(dist_chart_data_ref.index)

    # create plotly bar chart with two traces (dataset and reference)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dist_chart_data.index,
            y=dist_chart_data.values,
            name="Dataset",
            marker_color="#0068c9",
            # opacity=0.7,
        )
    )
    fig.add_trace(
        go.Bar(
            x=dist_chart_data_ref.index,
            y=dist_chart_data_ref.values,
            name="Reference",
            marker_color="#9ad3ff",
        )
    )

    fig.update_layout(
        barmode="group",
        xaxis_tickangle=-45,
        xaxis_title="Attribute Value Combinations",
        yaxis_title="Occurrence Percentage (%)",
        legend_title="Dataset",
    )
    fig.update_layout(xaxis_rangeslider_visible=True, xaxis_range=[-0.5, 10])
    # overlayed plotly bar chart
    st.plotly_chart(
        fig,
        use_container_width=True,
        use_container_height=True,
    )


@st.cache_resource
def show_true_diversity(shannon_divs, max_even_shannon_divs, entropy_cols, y_length):
    shannon_divs = dict(
        sorted(
            shannon_divs.items(),
            key=lambda item: max_even_shannon_divs[item[0]],
            reverse=True,
        )
    )
    # make sure that max_even_shannon_divs has keys in the same order as shannon_divs
    max_even_shannon_divs = {k: max_even_shannon_divs[k] for k in shannon_divs.keys()}

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(shannon_divs.keys()),
            y=list(shannon_divs.values()),
            name="Dataset",
            marker_color="#0068c9",
            error_y=dict(
                type="data",
                array=list(max_even_shannon_divs.values()),
                arrayminus=list(shannon_divs.values()),
                visible=True,
                symmetric=False,
                width=5,
                color="#0068c9",
            ),
            # add hovertext to display error_y values but sum y and error_y for Max Even Distribution
            hovertext=[
                f"True Diversity: {round(y, 3)}<br>Max Even Distribution: {round(y+error_y, 0)}"
                for y, error_y in zip(
                    shannon_divs.values(), max_even_shannon_divs.values()
                )
            ],
            hoverinfo="text",
        )
    )
    # fig.add_trace(
    #     go.Bar(
    #         x=list(max_even_shannon_divs.keys()),
    #         y=list(max_even_shannon_divs.values()),
    #         name="Max Even Distribution",
    #         marker=dict(color="rgba(0, 0, 0, 0)"),  # Set the color as transparent
    #         # Set the border color to black
    #         marker_line=dict(color="darkblue", width=1),
    #     )
    # )

    fig.update_layout(
        barmode="group",
        xaxis_tickangle=-45,
        xaxis_title=entropy_cols[0],
        yaxis_title="1-Order True Diversity regarding " + entropy_cols[1],
        legend_title="Dataset",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
    )


@st.cache_resource
def show_true_diversity_with_ref(
    shannon_divs,
    max_even_shannon_divs,
    shannon_divs_ref,
    max_even_shannon_divs_ref,
    entropy_cols,
):
    shannon_divs = dict(
        sorted(
            shannon_divs.items(),
            key=lambda item: max_even_shannon_divs[item[0]],
            reverse=True,
        )
    )
    shannon_divs_ref = dict(
        sorted(
            shannon_divs_ref.items(),
            key=lambda item: max_even_shannon_divs_ref[item[0]],
            reverse=True,
        )
    )
    # make sure that max_even_shannon_divs have keys in the same order as shannon_divs
    max_even_shannon_divs = {k: max_even_shannon_divs[k] for k in shannon_divs.keys()}
    max_even_shannon_divs_ref = {
        k: max_even_shannon_divs_ref[k] for k in shannon_divs_ref.keys()
    }
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(shannon_divs.keys()),
            y=list(shannon_divs.values()),
            name="Dataset",
            marker_color="#0068c9",
            error_y=dict(
                type="data",
                array=list(max_even_shannon_divs.values()),
                arrayminus=list(shannon_divs.values()),
                visible=True,
                symmetric=False,
                width=5,
                color="#0068c9",
            ),
        )
    )

    fig.add_trace(
        go.Bar(
            x=list(shannon_divs_ref.keys()),
            y=list(shannon_divs_ref.values()),
            name="Reference",
            marker_color="#9ad3ff",
            error_y=dict(
                type="data",
                array=list(max_even_shannon_divs_ref.values()),
                arrayminus=list(shannon_divs_ref.values()),
                visible=True,
                symmetric=False,
                width=5,
                color="#9ad3ff",
            ),
        )
    )
    # fig.add_trace(
    #     go.Bar(
    #         x=list(max_even_shannon_divs_ref.keys()),
    #         y=list(max_even_shannon_divs_ref.values()),
    #         name="Max Even Distribution (Reference)",
    #         marker=dict(color="rgba(0, 0, 0, 0)"),  # Set the color as transparent
    #         # Set the border color to black
    #         marker_line=dict(color="lightblue", width=1),
    #     )
    # )

    fig.update_layout(
        barmode="group",
        xaxis_tickangle=-45,
        xaxis_title=entropy_cols[0],
        yaxis_title="1-Order True Diversity regarding " + entropy_cols[1],
        legend_title="Dataset",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
    )


@st.cache_resource
def stplot_hill_numbers(occurences_general_i):
    plot_data = {}
    for k, v in occurences_general_i.items():
        hill_numbers = entropy.calc_hill_numbers(v, list(np.arange(0, 3.1, 0.1)))
        plot_data[", ".join(k)] = hill_numbers
        # add uniform distribution to plot as reference
        plot_data["uniform dist_" + ", ".join(k)] = [len(v)] * len(hill_numbers)
    st.plotly_chart(
        px.line(
            pd.DataFrame(plot_data, index=list(np.arange(0, 3.1, 0.1))),
            labels={
                "index": "q",
                "value": "True Diversity",
            },
        ),
        theme="streamlit",
        use_container_width=True,
    )


# def update_form():
