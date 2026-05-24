import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_best_model(X_train, y_train, X_test, y_test):
    model_configs = {
        'XGBoost': {
            'model': XGBClassifier(tree_method='hist', eval_metric='logloss', random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [200, 500],
                'max_depth': [6, 10],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 300],
                'max_depth': [10, 20],
                'min_samples_split': [5, 10],
                'bootstrap': [True, False]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5, 8],
                'subsample': [0.8, 1.0]
            }
        }
    }

    results = []
    optimized_models = {}

    print(f'{"Model":<18} | {"Acc CV":<10} | {"Acc Test":<10} | {"F1 Test":<10}')
    print('-' * 60)

    for name, config in model_configs.items():
        search = RandomizedSearchCV(
            estimator=config['model'],
            param_distributions=config['params'],
            n_iter=2,
            scoring='accuracy',
            cv=3,
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        
        results.append({
            'Model': name,
            'Acc CV': search.best_score_,
            'Acc Test': acc,
            'F1 Test': f1,
            'Precision Test': prec,
            'Recall Test': rec,
            'Best Params': search.best_params_
        })
        
        optimized_models[name] = best_model
        print(f'{name:<18} | {search.best_score_:<10.4f} | {acc:<10.4f} | {f1:<10.4f}')

    df_results = pd.DataFrame(results).sort_values(by='Acc Test', ascending=False)
    return df_results, optimized_models