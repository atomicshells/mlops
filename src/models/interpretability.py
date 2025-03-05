import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(model, features):
    """
    Plots feature importance for the given model.

    Args:
        model: Trained model with feature_importances_ attribute.
        features (list): List of feature names.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(12, 5))
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Importance')
    plt.show()
