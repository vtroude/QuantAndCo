import os
import shap
import pickle

import numpy    as np
import pylab    as pl
import pandas   as pd

from joblib  import load

from DataPipeline.get_data                  import get_all_data
from Model.Classifier.randomforest_barrier  import get_model_path

def get_shapley_path(model_name, data_type="values"):
    return f"Shapley/{model_name}_{data_type}.pkl"

def get_shapley(model_name, data):
    shapley_path    = get_shapley_path(model_name, "values")
    explainer_path  = get_shapley_path(model_name, "explainer")
    if os.path.exists(shapley_path):
        with open(explainer_path, 'rb') as f:
            explainer   = pickle.load(f)
        with open(shapley_path, 'rb') as f:
            shap_values = pickle.load(f)
    else:
        model       = load(get_model_path(model_name))
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)

        with open(explainer_path, 'wb') as f:
            pickle.dump(explainer, f)
        with open(shapley_path, "wb") as f:
            pickle.dump(shap_values, f)
    
    return shap_values, explainer

def get_top_features(shap_values, data, n=5):
    # Get the mean absolute SHAP values for each feature
    shap_sum    = np.abs(shap_values[1]).mean(axis=0)
    importance_df   = pd.DataFrame([data.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ['Feature', 'SHAP Importance']
    importance_df   = importance_df.sort_values('SHAP Importance', ascending=False)

    # Get the names of the top 5 features
    return importance_df['Feature'].head(n).values

def get_all_features_importance(model_name, data):
    # Using SHAP to explain feature importances
    shap_values, explainer  = get_shapley(model_name, data)

    # Create a single figure with subplots
    fig, axs = pl.subplots(2, 3, figsize=(20, 20))
    axs = axs.flatten()

    # Summary plot on the first subplot
    pl.sca(axs[0])
    shap.summary_plot(shap_values, data, plot_type="bar", show=False)
    pl.sca(axs[1])
    shap.summary_plot(shap_values[1], data, show=False)

    # Get the mean absolute SHAP values for each feature
    top_4_features  = get_top_features(shap_values, data, n=4)

    # Plot the SHAP dependence plots for the top 5 features
    for i, feature_name in enumerate(top_4_features, 2):
        pl.sca(axs[i])
        shap.dependence_plot(feature_name, shap_values[1], data, display_features=data, ax=axs[i], show=False)

    pl.show()

    random_indices = np.random.choice(data.index, 3, replace=False)

    # Force plots for three random predictions
    for i in random_indices:
        shap.force_plot(explainer.expected_value[1], shap_values[1][data.index.get_loc(i)], data.iloc[data.index.get_loc(i)])   #, matplotlib=True)

def get_interaction(model_name, data):
    # Using SHAP to explain feature importances
    shap_values, explainer  = get_shapley(model_name, data)

    # Convert SHAP values for positive class to DataFrame (if binary classification)
    #shap_df = pd.DataFrame(shap_values[1], columns=data.columns)

    # Compute correlation matrix for SHAP values
    #shap_corr = shap_df.corr()

    random_indices  = np.random.choice(data.index, size=500, replace=False)
    sampled_data    = data.loc[random_indices]
    # Compute SHAP interaction values
    shap_interaction_values = explainer.shap_interaction_values(sampled_data)
    # If binary classification, select the interaction values for the positive class
    #shap_interaction_values = np.mean(shap_interaction_values[1], axis=0)

    # Convert to DataFrame
    #shap_interaction_df = pd.DataFrame(
    #    shap_interaction_values,
    #    columns=sampled_data.columns,
    #    index=sampled_data.columns
    #)

    # Get the mean absolute SHAP values for each feature
    top_4_features  = get_top_features(shap_values, data, n=4)
    print(top_4_features)

    # Convert the mean interaction values DataFrame to a numpy array
    #shap_interaction_array = shap_interaction_df.values

    # Use the summary plot function for interaction values
    shap.summary_plot(shap_interaction_values[0], sampled_data)

    # Plotting interaction values can be very insightful
    # For example, you can plot the interaction values for a specific feature
    #for feature_name in top_4_features:
    #    shap.summary_plot(shap_interaction_df[feature_name].values, sampled_data)

    # Alternatively, you can also visualize the interaction between two specific features
    shap.dependence_plot(
        top_4_features[:2],
        shap_interaction_values[0],
        sampled_data
    )

if __name__ == "__main__":
    np.random.seed(42)

    date1   = "2021-04-05-23:44:12"
    date2   = "2024-04-04-23:44:12"
    time    = pd.to_datetime("2024-01-01 00:00:00")

    symbol      = 'BTCUSDT'
    interval    = ['1m', '1h', '1d', '1w']

    n_points    = 60

    print("Collect Data")
    data    = get_all_data(symbol, interval, date1, date2).dropna()
    data    = data[data.index > time]
    print("Data Collected")

    #get_all_features_importance("rf_classifier_hitting", data)
    get_interaction("rf_classifier_hitting", data)
    #get_all_features_importance("rf_classifier_direction", data)