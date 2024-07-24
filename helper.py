import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport, compare
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display_html, display_markdown, HTML, Markdown as md
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.lines import Line2D
import math
import re
from scipy import stats
import pickle
from joblib import dump
import time
from typing import Union

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import KFold, GridSearchCV


NUM_DECIMAL_PLACES = 7

def load_data(file_name: str,
              dropped_columns: list[str],
              na_value_representations: list[str]) -> pd.DataFrame:
    """
    Loads in user's input file as a pandas DataFrame and converts various representations of missing
    values to NaN. The file should be stored in the 'data' directory.
    
    Args:
        file_name (str): Name of file containing data for clustering
        dropped_columns (list[str]): List of columns to drop from the dataframe
        na_value_representations (list[str]): List of strings that represent missing values in
            the dataset
    Returns:
        df (pd.DataFrame): Dataframe of variable values for all data entries
    """
    # Automatically prepends 'data/' to the file name
    file_name: str = "data/" + file_name
    file_extension: str = file_name.split(".")[-1]

    if file_extension == "csv":
        df = pd.read_csv(file_name)
    elif file_extension in ["xls", "xlsx"]:
        if file_extension == "xls":
            df = pd.read_excel(file_name, engine = 'xlrd')
        else:
            df = pd.read_excel(file_name, engine = 'openpyxl')
    elif file_extension == "json":
        df = pd.read_json(file_name)
    else:
        raise ValueError("""Unsupported file format or misspelled file name. Please upload 
                         a CSV, Excel, or JSON file and ensure the file name is spelled correctly.""")
    
    # Replaces input representations of missing values with np.nan
    df = df.replace(na_value_representations, np.nan)
    
    df = df.drop_duplicates()
    df = df.drop(columns = dropped_columns)
    
    return df


def get_unique_categories(df: pd.DataFrame, categorical_columns: list[str], show_category_count: bool = True, show_categories: bool = False) -> dict[str, int]:
    """
    Creates a dictionary of the unique categories in each categorical column of the input DataFrame and displays
    the number of unique categories for each varaible. Optionally prints the unique categories.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        categorical_columns (list[str]): A list of column names representing categorical variables.
        show_category_count (bool): If True, displays the number of unique categories for each categorical variable.
        show_categories (bool): If True, displays the unique categories for each categorical variable.

    Returns:
        dict[str, list[str]]: A dictionary where keys are categorical variable column names and values are a list of unique categories.
    """
    categoris_dict: dict = {}
    for variable in categorical_columns:
        categoris_dict[variable] = list(df[variable].unique())
    
    if show_category_count:
        display_text("Number of Unique Categories in Categorical Columns:", font_size = 20)
        for variable, categories in categoris_dict.items():
            display_text(f"Number of Unique Categories in '{variable}': {len(categories)}")
        print()
        
    if show_categories:
        for variable, categories in categoris_dict.items():
            display_text(f"Unique Categories in '{variable}': {categories}")
        print()
    
    return categoris_dict


def display_df(df: pd.DataFrame,
                      font_size: int = 14) -> None:
    """
    Displays the passed in DataFrame with the specified font size.

    Args:
        df (pd.DataFrame): The DataFrame to be displayed.
        font_size (int): The font size at which the items in the
        DataFrame should be displayed.

    Returns:
        None
    """
    df_html = df.to_html()
    styled_html = f'<div style="font-size: {font_size}px;">{df_html}</div>'
    display_html(HTML(styled_html))
    

def display_text(text: str,
                 font_size: int = 16,
                 font_weight = 'normal') -> None:
    """
    Displays the passed in text with the specified font size and font weight.

    Args:
        text (str): The text to be displayed.
        font_size (int): The font size at which the text should be displayed.
        font_weight: The font weight (e.g., 'normal', 'bold', 'bolder', 'lighter',
        or numeric value from 100 to 900).

    Returns:
        None
    """
    styled_html = f'<div style="font-size: {font_size}px; font-weight: {font_weight};">{text}</div>'
    display_html(HTML(styled_html))
    

def string_to_float(value_str: str):
    """
    Cleans a string by removing all characters except digits and decimal points directly followed by a digit
    using regular expressions and converts the cleaned string to a float. This function may not work as
    intended with some strings that contain multiple and/or awkwardly placed decimal points.

    Args:
        value_str (str): The string to be cleaned and converted.

    Returns:
        float or None: The cleaned float value or None if conversion fails.
    """
    # Check if the value is np.nan and returns None if it is
    if pd.isna(value_str):
        return None
    
    # Remove all characters except digits and decimal points followed by a digit.
    cleaned_str: str = re.sub(r'[^0-9.]+', '', value_str)
    
    # Additional check to handle multiple decimal points or trailing decimal points
    parts: list[str] = cleaned_str.split('.')
    if len(parts) > 2:
        cleaned_str: str = ''.join(parts[:-1]) + '.' + parts[-1]
    elif len(parts) == 2 and parts[1] == '':
        cleaned_str: str = parts[0]
    
    # Attempt to convert string to float
    try:
        float_value = pd.to_numeric(cleaned_str)
    except ValueError:
        print(f"Failed to convert {value_str} to a float")
        # Sets the float_value to None if the string cannot be converted to a float
        float_value = None
    
    return float_value


def drop_rows_with_missing_values(df: pd.DataFrame,
                                  columns_to_check: list[str]) -> pd.DataFrame:
    """
    Makes a copy of the input DataFrame and drops rows that have one or more missing values in any of the columns specified by 
    the columns_to_check parameter (does not mutate the input DataFrame). Also prints the number of entries dropped and the
    resulting total number of entries.
    
    Args:
        df (pd.DataFrame): DataFrame containing loded in data
        columns_to_check (list[str]): List of columns to check for missing values
    Returns:
        dropna_df (pd.DataFrame): DataFrame with missing values dropped
    """
    
    original_number_of_entries = len(df)
    
    dropna_df = df.dropna(subset = columns_to_check)
    new_number_of_entries = len(dropna_df)
    number_of_entries_dropped = original_number_of_entries - new_number_of_entries
    
    display_text(f"drop_rows_with_missing_values Results: {number_of_entries_dropped} Entries Dropped") 
    display_text(f"New Number of Entries: {new_number_of_entries}")
    
    return dropna_df


def impute_missing_values(df: pd.DataFrame,
                          numerical_columns_to_impute: list[str],
                          categorical_columns_to_impute: list[str]) -> pd.DataFrame:
    """
    Imputes missing values in the DataFrame with either the median value (for numerical variables) or the most frequent value
    (for categorical variables).
    
    Args:
        numerical_columns_to_impute (list[str]): List of the names of numerical columns with missing values to impute
        categorical_columns_to_impute (list[str]): List of the names of categorical columns with missing values to impute
    Returns:
        impute_df (pd.DataFrame): DataFrame with missing values imputed
    """
    impute_df = df.copy()
    
    # Here is where to configure the imputation strategy if need be
    numerical_imputer = SimpleImputer(strategy = "median")
    categorical_imputer = SimpleImputer(strategy = "most_frequent")
    
    impute_df[numerical_columns_to_impute] = numerical_imputer.fit_transform(impute_df[numerical_columns_to_impute])
    impute_df[categorical_columns_to_impute] = categorical_imputer.fit_transform(impute_df[categorical_columns_to_impute])
    
    display_text("Missing Values Successfully Imputed")
    
    return impute_df


def visualize_outliers(df: pd.DataFrame,
                       numerical_columns_to_check: Union[list[str], str],
                       iqr_multiplier: float = 1.5,
                       display: bool = True,
                       remove: bool = False,
                       remove_option: str = "both") -> pd.DataFrame:
    """
    Creates a boxplot for each column of the input Dataframe in the numerical_columns_to_check parameter to help users visualize potential
    outliers in their dataset. Below this boxplot, the function prints the number of high and low outliers (determined by the IQR method) in the
    current column. The upper and lower bounds for outliers are denoted by red dotted lines. Points below the low bound red dotted line
    or above the high bound red dotted line are consideered outliers. Users can choose whether to drop outlier entries through the remove
    boolean parameter. You can change which points are considered outliers by changing the iqr_multiplier parameter.
    
    The lower and upper whiskers of the boxplot denote the 5th and 95th percentile of the current column's values respectively.

    This function can be used iteratively to handle outliers in different columns with varying sensitivity levels. It allows for
    selective removal of entries below/above the red dotted lines. The function can be run without displaying visualizations for efficiency.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be analyzed.
        numerical_columns_to_check (Union[list[str], str], default = numerical_variables): List of the names of columns to check for outliers. The
            default argument is a list of all numerical columns in the input DataFrame.
        iqr_multiplier (float, default = 1.5): Multiplier for the IQR to define the outlier threshold. Higher values are more lenient, increasing
            the range of the upper and lower red dotted lines. Lower values are more strict, decreasing the range of the red dotted lines.
        display (bool, default = True): If True, displays boxplots for each variable. If false, only outlier statistics are printed.
        remove (bool, default = False): If True, removes identified outliers from the DataFrame.
        remove_option (str, default = 'both'): Specifies which outliers to remove: 'both' removes all identified outliers, 'upper' only removes
            outliers greater than the upper bound (values past the upper red dotted line), and 'lower' only removes outliers less than the
            lower bound (values behind the lower red dotted line). This parameter has no effect if remove = False.

    Returns:
        pd.DataFrame: The original DataFrame if remove = False, otherwise a new DataFrame with outliers removed.
    """
    # If a single column is passed in as a string, convert it to a list so the following for loop still works properly
    if type(numerical_columns_to_check) == str:
        numerical_columns_to_check = [numerical_columns_to_check]
        
    for col in numerical_columns_to_check:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (iqr * iqr_multiplier)
        upper_bound = q3 + (iqr * iqr_multiplier)
        
        # Only create plot if display is True
        if display:
            plt.figure(figsize = (10, 6))
            ax = sns.boxplot(x = df[col], whis = [5, 95])
            plt.title(f'Boxplot of {col}')
            
            # Add vertical red dotted lines for lower and upper bounds if within the plot's x-axis limits
            x_min, x_max = ax.get_xlim()
            if x_min <= lower_bound <= x_max:
                plt.axvline(lower_bound, color='red', linestyle='dotted', linewidth=1)
            if x_min <= upper_bound <= x_max:
                plt.axvline(upper_bound, color='red', linestyle='dotted', linewidth=1)
            
            # Create legend
            legend_lines = [Line2D([0], [0], color='red', linestyle='dotted', linewidth=1)]
            legend_labels = ['Lower/Upper Bound']
            plt.legend(legend_lines, legend_labels, loc='upper right')
            plt.show()
        
        lower_outlier_count = df[col][df[col] < lower_bound].count()
        upper_outlier_count = df[col][df[col] > upper_bound].count()
        
        display_text(f"{col}:", font_size = 18, font_weight = 'bold')
        display_text(f"- Lower Bound for Outliers: {lower_bound}", font_size = 16)
        display_text(f"- Upper Bound for Outliers: {upper_bound}", font_size = 16)
        display_text(f"- Number of Outliers Below Lower Bound: {lower_outlier_count}", font_size = 16)
        display_text(f"- Number of Outliers Above Upper Bound: {upper_outlier_count}", font_size = 16)
        print()
        
    # Removes outliers from the DataFrame if remove = True
    if remove:
        # Calculate indices of outliers
        lower_outlier_indices = df.index[df[col] < lower_bound].tolist()
        upper_outlier_indices = df.index[df[col] > upper_bound].tolist()
        outlier_indices_to_be_removed = set()
        
        # Add outlier indices that will be removed to outlier_indices_to_be_removed based on the remove_option parameter
        if remove_option == "both":
            outlier_indices_to_be_removed.update(lower_outlier_indices)
            outlier_indices_to_be_removed.update(upper_outlier_indices)
        elif remove_option == "lower":
            outlier_indices_to_be_removed.update(lower_outlier_indices)
        elif remove_option == "upper":
            outlier_indices_to_be_removed.update(upper_outlier_indices)
        else:
            raise ValueError("Invalid argument passed into remove_option parameter. Please use 'both', 'lower', or 'upper'.")
            
        removed_outliers_df = df.drop(index = outlier_indices_to_be_removed)
        display_text(f"Total Number of Outlier Entries Removed in {col}: {len(outlier_indices_to_be_removed)}", font_size = 18)
        print()
        return removed_outliers_df
    
    # Simply return the original DataFrame if remove = False
    return df


def create_model_pipelines(
    models: list[BaseEstimator],
    preprocessor: Union[ColumnTransformer, BaseEstimator, list[Union[ColumnTransformer, BaseEstimator]]]
) -> list[BaseEstimator]:
    """
    Creates a list of pipelines, each combining the preprocessor(s) with a model.
    Works with ColumnTransformer and other scikit-learn preprocessors.

    Args:
        models (list[BaseEstimator]): A list of instantiated model objects.
        preprocessor (Union[ColumnTransformer, BaseEstimator, list[Union[ColumnTransformer, BaseEstimator]]]):
            A single preprocessor (ColumnTransformer or other) or a list of preprocessors.

    Returns:
        list[BaseEstimator]: A list of pipeline objects, each combining the preprocessor(s) with a model.
    """
    pipelines = []

    # Ensure preprocessor is a list
    if not isinstance(preprocessor, list):
        preprocessor_list = [preprocessor]

    for model in models:
        # If there's only one preprocessor and it's a ColumnTransformer,
        # we don't need to wrap it in another make_pipeline call
        if len(preprocessor_list) == 1 and isinstance(preprocessor_list[0], ColumnTransformer):
            pipeline = make_pipeline(preprocessor_list[0], model)
        else:
            # For multiple preprocessors or non-ColumnTransformer preprocessors,
            # we use make_pipeline to combine them
            pipeline = make_pipeline(*preprocessor_list, model)
        
        pipelines.append(pipeline)

    return pipelines


def get_clustering_model(model):
    """Extract the clustering model from a pipeline or return the model if it's not a pipeline."""
    if isinstance(model, Pipeline):
        return model.steps[-1][1]  # Last step of the pipeline
    return model


def generate_model_key(model, index):
    """Generate a key for the model, handling both pipelines and direct models."""
    clustering_model = get_clustering_model(model)

    if hasattr(clustering_model, 'n_clusters'):
        return clustering_model.n_clusters
    elif hasattr(clustering_model, 'n_components'):
        return clustering_model.n_components
    else:
        return index


def fit_clustering_models(
    df: pd.DataFrame,
    models: list,
    model_name: str,
    custom_metric_func = None) -> dict[str, dict]:
    """
    Fits multiple clustering models, computes relevant metrics, and saves cluster sizes.

    Args:
        data (pd.DataFrame): The preprocessed input data as a numpy array.
        models (list): A list of instantiated clustering model objects or pipelines.
        custom_metric_func: A function that takes a fitted model and data, and returns a dictionary of custom metrics.

    Returns:
        dict[str, dict]: A dictionary where keys are model identifiers and values are dictionaries containing:
            - 'model': The fitted clustering model
            - 'labels': Cluster labels for each data point
            - 'train_fit_time': Time taken to train and fit the model
            - 'n_clusters': Number of clusters
            - 'cluster_sizes': Dictionary of cluster sizes (label: size)
            - 'silhouette': The silhouette score of the clustering
            - 'calinski_harabasz': The Calinski-Harabasz score of the clustering
            - 'davies_bouldin': The Davies-Bouldin score of the clustering
            - Any additional metrics returned by custom_metric_func
    """
    results = {}
    for idx, model in enumerate(models):
        # Fit the model and measure training time
        start_time = time.time()
        model.fit(df)

        # Get cluster labels
        if hasattr(model, 'labels_'):
            labels = model.labels_
        else:
            labels = model.predict(df)
        
        end_time = time.time()
        train_fit_time = end_time - start_time

        unique_labels, counts = np.unique(labels, return_counts = True)
        size_order = np.argsort(counts)[::-1]
        label_map = {old: new for new, old in enumerate(unique_labels[size_order])}
        new_labels = np.array([label_map[label] for label in labels])
        
        # Update the results with new labels
        cluster_sizes = dict(zip(range(len(counts)), counts[size_order]))
        n_clusters = len(cluster_sizes)

        # Prepare result dictionary
        model_data = {
            'model': model,
            'labels': new_labels,
            'train_time': train_fit_time,
            'n_clusters': n_clusters,
            'cluster_sizes': cluster_sizes
        }

        # Compute scoring metrics if there's more than one cluster
        if n_clusters > 1:
            model_data['silhouette'] = silhouette_score(df, labels)
            model_data['calinski_harabasz'] = calinski_harabasz_score(df, labels)
            model_data['davies_bouldin'] = davies_bouldin_score(df, labels)
        else:
            model_data['silhouette'] = np.nan
            model_data['calinski_harabasz'] = np.nan
            model_data['davies_bouldin'] = np.nan


        # Apply custom metric function if provided
        if custom_metric_func:
            clustering_model = get_clustering_model(model)
            custom_metrics = custom_metric_func(clustering_model)
            model_data.update(custom_metrics)

        key = generate_model_key(model, idx)

        results[key] = model_data

    return results


def get_consistent_cluster_order(cluster_sizes: dict) -> list:
    """
    Returns a consistent order of clusters based on their sizes.

    Args:
        cluster_sizes (dict): A dictionary where keys are cluster labels and values are the sizes of each cluster.

    Returns:
        list: A list of cluster labels sorted in descending order based on their sizes.
    """
    return sorted(cluster_sizes.keys())


def plot_cluster_sizes(model_results: dict, model_idx: int) -> None:
    """
    Plots the sizes of each cluster with consistent ordering.

    This function creates a bar plot showing the size of each cluster. Clusters are ordered
    consistently based on their size, and each bar is labeled with its corresponding size.

    Args:
        model_results (dict): A dictionary containing the results of multiple clustering models.
        model_idx (int): The index of the specific model to plot from the model_results dictionary.

    Returns:
        None. The function displays the plot directly.
    """
    selected_model = model_results[model_idx]
    cluster_size_dict = selected_model['cluster_sizes']
    consistent_order = get_consistent_cluster_order(cluster_size_dict)
    cluster_sizes = [cluster_size_dict[cluster] for cluster in consistent_order]

    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(consistent_order)))
    bars = plt.bar(range(len(consistent_order)), cluster_sizes, color=colors)
    plt.title('Cluster Sizes', fontweight='bold')
    plt.xlabel('Cluster')
    plt.xticks(range(len(consistent_order)), [f'Cluster {c}' for c in consistent_order])
    plt.ylabel('Number of Data Points')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}',
                 ha='center', va='bottom')
    
    plt.show()


def plot_feature_importance_heatmap(results: dict, model_idx: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates feature importance for each cluster and plots it as a heatmap with consistent cluster ordering.

    This function computes the importance of each feature for every cluster by comparing
    the cluster mean to the overall mean. It then visualizes this information as a heatmap.

    Args:
        results (dict): A dictionary containing the results of multiple clustering models.
        model_idx (int): The index of the specific model to plot from the model_results dictionary.
        df (pd.DataFrame): The original dataframe used for clustering.

    Returns:
        pd.DataFrame: A dataframe containing the feature importance scores for each cluster.
    """
    selected_model = results[model_idx]
    cluster_labels = selected_model['labels']
    
    data_with_clusters = df.copy()
    data_with_clusters['Cluster'] = cluster_labels
    
    overall_mean = data_with_clusters.drop('Cluster', axis=1).mean()

    feature_importance = {}
    for cluster in data_with_clusters['Cluster'].unique():
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster].drop('Cluster', axis=1)
        cluster_mean = cluster_data.mean()
        importance = abs(cluster_mean - overall_mean) / overall_mean
        feature_importance[cluster] = importance

    consistent_order = get_consistent_cluster_order(selected_model['cluster_sizes'])
    feature_importance_df = pd.DataFrame({f'Cluster {c}': feature_importance[c] for c in consistent_order})

    plt.figure(figsize=(12, 8))
    sns.heatmap(feature_importance_df, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Feature Importance by Cluster', fontweight='bold')
    plt.xlabel('Cluster')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    return feature_importance_df


def plot_feature_distributions(model_results: dict, model_idx: int, df: pd.DataFrame) -> None:
    """
    Plots box plots for each feature across clusters with consistent ordering.

    This function creates a grid of box plots, one for each feature in the dataset.
    Each box plot shows the distribution of a feature across different clusters.

    Args:
        model_results (dict): A dictionary containing the results of multiple clustering models.
        model_idx (int): The index of the specific model to plot from the model_results dictionary.
        df (pd.DataFrame): The original dataframe used for clustering.

    Returns:
        None. The function displays the plot directly.
    """
    selected_model = model_results[model_idx]
    cluster_labels = selected_model['labels']
    
    data_with_clusters = df.copy()
    data_with_clusters['Cluster'] = cluster_labels
    
    consistent_order = get_consistent_cluster_order(selected_model['cluster_sizes'])
    cluster_order = [f'Cluster {c}' for c in consistent_order]
    data_with_clusters['Cluster'] = data_with_clusters['Cluster'].map({c: f'Cluster {c}' for c in consistent_order})
    
    n_rows = math.ceil(len(df.columns) / 2)
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(20, 5*n_rows))
    fig.suptitle('Feature Distributions by Cluster', fontsize=16, fontweight='bold')
    
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, feature in enumerate(df.columns):
        ax = axes[i]
    
        sns.boxplot(x='Cluster', y=feature, data=data_with_clusters, ax=ax, order=cluster_order)
        ax.set_title(feature, fontweight='bold')
        ax.set_xlabel('Cluster')
        
        # Set tick positions explicitly
        ax.set_xticks(range(len(cluster_order)))
        # Then set tick labels with rotation
        ax.set_xticklabels(cluster_order, rotation=45, ha='right')
    
    # Remove unused subplots
    for i in range(len(df.columns), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.subplots_adjust(hspace=0.45, wspace=0.15)
    plt.show()


def plot_cluster_kde(model_results: dict, model_idx: int, df: pd.DataFrame) -> None:
    """
    Plots KDE (Kernel Density Estimation) plots for each feature across clusters with consistent ordering.

    This function creates a grid of KDE plots, one for each feature in the dataset.
    Each plot shows the probability density distribution of a feature for each cluster.

    Args:
        model_results (dict): A dictionary containing the results of multiple clustering models.
        model_idx (int): The index of the specific model to plot from the model_results dictionary.
        df (pd.DataFrame): The original dataframe used for clustering.

    Returns:
        None. The function displays the plot directly.
    """
    selected_model = model_results[model_idx]
    cluster_labels = selected_model['labels']
    
    data_with_clusters = df.copy()
    data_with_clusters['Cluster'] = cluster_labels
    
    consistent_order = get_consistent_cluster_order(selected_model['cluster_sizes'])
    
    fig, axes = plt.subplots(math.ceil(len(df.columns)/2), 2, figsize=(20, 6*math.ceil(len(df.columns)/2)))
    fig.suptitle('Feature Probability Distributions by Cluster', fontsize=16, fontweight='bold')
    axes = axes.flatten()  # Flatten for easy indexing
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(consistent_order)))
    
    for i, feature in enumerate(df.columns):
        for j, cluster in enumerate(consistent_order):
            cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster]
            sns.kdeplot(data=cluster_data[feature], ax=axes[i], label=f'Cluster {cluster}', color=colors[j], warn_singular=False)
    
        axes[i].set_title(f'{feature} Distribution by Cluster', fontweight="bold")
        axes[i].legend()
    
    for i in range(len(df.columns), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.975])
    plt.subplots_adjust(hspace=0.25, wspace=0.15)
    plt.show()
    
    
def analyze_clusters_with_profiling(df: pd.DataFrame, model_results: dict, model_idx: int, dataset_report: ProfileReport) -> dict:
    """
    Generate and display a pandas profiling report for each cluster in the Jupyter notebook.

    Args:
        df (pd.DataFrame): The original dataframe used for clustering.
        model_results (pd.DataFrame): Results of the clustering models
        n_clusters (int): The number of clusters
        dataset_report (ProfileReport): The pandas profiling report for the original dataset
    
    Returns:
        dict: A dictionary where keys are cluster labels and values are tuples containing the ProfileReport and ComparisonReport
        for each cluster.
    """
    selected_model = model_results[model_idx]
    cluster_labels = selected_model['labels']
    
    data_with_clusters = df.copy()
    data_with_clusters['Cluster'] = cluster_labels
    
    consistent_order = get_consistent_cluster_order(selected_model['cluster_sizes'])
    
    cluster_reports = {}
    for cluster in consistent_order:
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster].drop('Cluster', axis=1)
        report = ProfileReport(cluster_data, title=f"Cluster {cluster} Profiling Report", progress_bar = False, explorative = True)
        comparison_report = report.compare(dataset_report)
        cluster_reports[cluster] = (report, comparison_report)
        
    return cluster_reports


def plot_kmeans_analysis(results: dict, k_input: Union[range, list]):
    """
    Creates a comprehensive plot of K-Means clustering analysis results.

    This function generates a single figure with 4 subplots:
    1. Elbow curve
    2. Silhouette scores
    3. Calinski-Harabasz scores
    4. Davies-Bouldin scores

    Args:
        results (dict): A dictionary of K-Means results as returned by fit_kmeans_range().
        k_range (range): The range of k values used in the analysis.

    Returns:
        None: This function does not return anything, it only produces a plot.
    """
    k_values = list(k_input)
    inertias = [results[k]['inertia'] for k in k_values]
    silhouette_scores = [results[k]['silhouette'] for k in k_values]
    ch_scores = [results[k]['calinski_harabasz'] for k in k_values]
    db_scores = [results[k]['davies_bouldin'] for k in k_values]

    # Create a figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle('K-Means Clustering Analysis', fontsize=16, fontweight='bold')

    # Elbow curve
    axes[0, 0].plot(k_values, inertias, 'bo-')
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].set_title('Elbow Curve', fontweight='bold')

    # Silhouette scores
    axes[0, 1].plot(k_values, silhouette_scores, 'ro-')
    axes[0, 1].set_xlabel('Number of Clusters (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('Silhouette Scores', fontweight='bold')

    # Calinski-Harabasz scores
    axes[1, 0].plot(k_values, ch_scores, 'go-')
    axes[1, 0].set_xlabel('Number of Clusters (k)')
    axes[1, 0].set_ylabel('Calinski-Harabasz Score')
    axes[1, 0].set_title('Calinski-Harabasz Scores', fontweight='bold')

    # Davies-Bouldin scores
    axes[1, 1].plot(k_values, db_scores, 'mo-')
    axes[1, 1].set_xlabel('Number of Clusters (k)')
    axes[1, 1].set_ylabel('Davies-Bouldin Score')
    axes[1, 1].set_title('Davies-Bouldin Scores', fontweight='bold')

    # Adjust the layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.975])  # Adjust the rect parameter to accommodate the suptitle
    
    # Increase spacing between subplots
    plt.subplots_adjust(hspace=0.15, wspace=0.2)
    
    plt.show()

    
def save_model_pickle(file_name: str,
                      num_clusters: int,
                      results: dict[str, dict]) -> None:
    """
    Saves the best performing model to a pickle file.
    
    Args:
        file_name (str): Name of the pickle file to save the model to
        model_name (str): Name of the model that is about to be saved
        clustering_results (dict[str, dict]): Dictionary containing the model object, predictions on the testing data, and other model perfomance data
    Returns:
        None
    """
    model_to_save = results[num_clusters]["model"]
    with open(file_name, 'wb') as file:
        pickle.dump(model_to_save, file)
        

def save_model_joblib(file_name: str,
                      num_clusters: int,
                      results: dict[str, dict]) -> None:
    """
    Saves the best performing model to a joblib file.
    
    Args:
        file_name (str): Name of the joblib file to save the model to
        model_name (str): Name of the model that is about to be saved
        results (dict[str, dict]): Dictionary containing the model object, predictions on the testing data, and other model perfomance data
    Returns:
        None
    """
    model_to_save = results[num_clusters]["model"]
    dump(model_to_save, file_name)
    

def save_model(file_name: str,
               num_clusters: int,
               results: dict[str, dict],
               method: str) -> None:
    """
    Saves the best performing model to a file using the specified method.
    
    Args:
        file_name (str): Name of the file to save the model to
        model_name (str): Name of the model to save
        results (dict[str, dict]): Dictionary containing the model object, predictions on the testing data, and other model perfomance data
        method (str): Method to use for saving the model ("pickle" or "joblib")
    Returns:
        None
    """
    if method == "pickle":
        save_model_pickle(file_name, num_clusters, results)
    elif method == "joblib":
        save_model_joblib(file_name, num_clusters, results)
    else:
        raise ValueError("Invalid method specified. Please use 'pickle' or 'joblib'.")
    
    
