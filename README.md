# Diversity Profile for Structured Data

This repository contains the code for the thesis "Quantification of Diversity in Structured Data".
To visualize and facilitate the understanding and interpretation of a selected collection and application of diversity measures in the diversity dimensions of **Coverage** and **Heterogeneity**, a prototypical user interface (UI) for the diversity profile has been developed. This repository contains the code for this UI.

### Installation

This tool is a streamlit app. To run it, you need to install streamlit and further dependencies for the calculation of the diversity dimensions **Coverage** and **Heterogeneity**.

It is recommended to use a virtual environment for this.
You can do this either by using the `environment.yml` file.

#### Using environment.yml

```bash
conda env create -f environment.yml
```

### Usage

To run the app, simply execute the following command:

```bash
streamlit run diversity_profile.py
```

The app will open in your browser. The main code is located in `diversity_profile.py`. The code for the visualization of the diversity dimensions is located in `src/st_output.py`.
Further calculation on the coverage calculation is imported from the `src/coverage.py` file.
The frequency profiling is done in `src/frequency_profiling.py`.

The app currently includes a selection of pre-defined datasets which are stored in the `data` folder. You can add your own datasets by adding them to the `data` folder and adding them to the `datasets` list in `st_output.py`.
The datasets 'ACSIncome' and 'UKRoadSafety' that are mentioned in the thesis can be reproduced by running the `0_create_datasets_pipeline.ipynb` script in the `notebooks` folder.
