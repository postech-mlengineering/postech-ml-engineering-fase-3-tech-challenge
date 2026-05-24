from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report



def tune_xgboost(X_train, y_train, X_test, y_test, n_iter=10):
    '''
    Realiza o ajuste de hiperparâmetros específico para o XGBoost.
    '''
    xgb = XGBClassifier(
        tree_method='hist',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    params = {
        'n_estimators': [100, 300, 500, 1000],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5],
        'min_child_weight': [1, 3, 5]
    }
    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=params,
        n_iter=n_iter,
        scoring='f1_weighted',
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    print('Iniciando o ajuste de hiperparâmetros para o XGBoost.')
    search.fit(X_train, y_train)

    best_xgb = search.best_estimator_
    y_pred = best_xgb.predict(X_test)

    metrics = {
        'Acc Test': accuracy_score(y_test, y_pred),
        'F1 Test': f1_score(y_test, y_pred, average='weighted'),
        'Precisão': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'Best CV Score': search.best_score_
    }

    print('\n' + '='*30)
    print('Melhores parâmetros encontrados:')
    for param, value in search.best_params_.items():
        print(f'{param}: {value}')
    
    print('\nResultados:')
    for metric, val in metrics.items():
        print(f'{metric}: {val:.4f}')
    print('='*30)
    
    print('\nRelatório de Classificação:')
    print(classification_report(y_test, y_pred))

    return best_xgb, search.best_params_, metrics


def tune_random_forest(X_train, y_train, X_test, y_test, n_iter=10):
    '''
    Realiza o ajuste de hiperparâmetros específico para o Random Forest.
    '''
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    params = {
        'n_estimators': [100, 300, 500, 800],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=params,
        n_iter=n_iter,
        scoring='f1_weighted',
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    print('Iniciando o ajuste de hiperparâmetros para o Random Forest.')
    search.fit(X_train, y_train)

    best_rf = search.best_estimator_
    y_pred = best_rf.predict(X_test)

    metrics = {
        'Acc Test': accuracy_score(y_test, y_pred),
        'F1 Test': f1_score(y_test, y_pred, average='weighted'),
        'Precisão': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'Best CV Score': search.best_score_
    }

    print('\n' + '='*30)
    print('Melhores parâmetros (Random Forest):')
    for param, value in search.best_params_.items():
        print(f'{param}: {value}')
    
    print('\nResultados:')
    for metric, val in metrics.items():
        print(f'{metric}: {val:.4f}')
    print('='*30)
    
    print('\nRelatório de Classificação:')
    print(classification_report(y_test, y_pred))

    return best_rf, search.best_params_, metrics


def tune_gradient_boosting(X_train, y_train, X_test, y_test, n_iter=10):
    '''
    Realiza o ajuste de hiperparâmetros específico para o Gradient Boosting.
    '''
    gb = GradientBoostingClassifier(random_state=42)
    
    params = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 8],
        'subsample': [0.7, 0.8, 1.0],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }

    search = RandomizedSearchCV(
        estimator=gb,
        param_distributions=params,
        n_iter=n_iter,
        scoring='f1_weighted',
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    print('Iniciando o ajuste de hiperparâmetros para o Gradient Boosting.')
    search.fit(X_train, y_train)

    best_gb = search.best_estimator_
    y_pred = best_gb.predict(X_test)

    metrics = {
        'Acc Test': accuracy_score(y_test, y_pred),
        'F1 Test': f1_score(y_test, y_pred, average='weighted'),
        'Precisão': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'Best CV Score': search.best_score_
    }

    print('\n' + '='*30)
    print('Melhores parâmetros (Gradient Boosting):')
    for param, value in search.best_params_.items():
        print(f'{param}: {value}')
    
    print('\nResultados:')
    for metric, val in metrics.items():
        print(f'{metric}: {val:.4f}')
    print('='*30)
    
    print('\nRelatório de Classificação:')
    print(classification_report(y_test, y_pred))

    return best_gb, search.best_params_, metrics