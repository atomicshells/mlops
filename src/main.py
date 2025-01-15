from src.data.load_data import load_data
from src.features.build_features import build_features
from models.train_model import train_models, evaluate_models
from models.optimal_parameters import optimize_model
from models.feature_importance import plot_feature_importance

def main():
    data = load_data('data/data.csv')
    X, y = build_features(data)
    models, X_train, X_test, y_train, y_test = train_models(X, y)
    results = evaluate_models(models, X_test, y_test)
    print(results)
    best_model = optimize_model(models['AdaBoost'], {'n_estimators': [50, 100, 150]}, X_train, y_train)
    plot_feature_importance(best_model, X_train.columns)

if __name__ == '__main__':
    main()