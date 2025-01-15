import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(model, features):
    """
    Plots the feature importances of the given model.

    Args:
        model (model object): A trained model that supports feature importance.
        features (list): A list of feature names that correspond to the model features.

    Description:
        Displays a bar chart showing the importance of each feature in the model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(8, 6))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()