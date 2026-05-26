from typing import Dict, Any
import logging
import joblib

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, classification_report
)
from sklearn.model_selection import train_test_split

from src.feature_engineering import FeatureEngineer
from src.model_config import MODEL_CONFIG


logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, model_configs: Dict[str, Any] = MODEL_CONFIG):
        self.model_configs = model_configs
        self.results_df = None
        self.best_overall_model = None

    def _evaluate(self, model, X_test, y_test) -> Dict[str, float]:
        '''Calcula métricas de teste.'''
        y_pred = model.predict(X_test)
        return {
            'Acc Test': accuracy_score(y_test, y_pred),
            'F1 Test': f1_score(y_test, y_pred, average='weighted'),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted')
        }

    def run_model_selection(self, X_train, y_train, X_test, y_test, n_iter=2) -> str:
        '''
        Seleciona melhor modelo.
        '''
        logger.info('Iniciando seleção de modelos.')
        summary = []

        for name, config in self.model_configs.items():
            logger.info(f'Testando {name}')
            
            search = RandomizedSearchCV(
                estimator=config['model'],
                param_distributions=config['params'],
                n_iter=n_iter,
                scoring='accuracy',
                cv=3,
                random_state=42,
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            
            metrics = self._evaluate(search.best_estimator_, X_test, y_test)
            metrics['Model'] = name
            metrics['Best CV Score'] = search.best_score_
            summary.append(metrics)

        self.results_df = pd.DataFrame(summary).sort_values(by='F1 Test', ascending=False)
        best_model = self.results_df.iloc[0]['Model']
        
        print('\n--- Resultados ---')
        print(self.results_df[['Model', 'Best CV Score', 'F1 Test', 'Acc Test']])
        return best_model

    def run_fine_tuning(self, model, X_train, y_train, X_test, y_test, n_iter=10):
        '''
        Aplica fine-tuning para o modelo selecionado.
        '''
        logger.info(f'Iniciando fine-tuning para o modelo selecionado: {model}')
        
        config = self.model_configs[model]
        
        search = RandomizedSearchCV(
            estimator=config['model'],
            param_distributions=config['params'],
            n_iter=n_iter,
            scoring='f1_weighted',
            cv=5,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        search.fit(X_train, y_train)
        self.best_overall_model = search.best_estimator_
        
        y_pred = self.best_overall_model.predict(X_test)
        metrics = self._evaluate(self.best_overall_model, X_test, y_test)
        
        print('\n' + '='*40)
        print(f'Melhor modelo: {model}')
        print('='*40)
        print(f'Parâmetros: {search.best_params_}')
        print('\nRelatório de classificação:')
        print(classification_report(y_test, y_pred))
        
        return self.best_overall_model, search.best_params_, metrics


if __name__ == '__main__':
    df = pd.read_pickle('data/curated/features.pkl')

    X = df.drop(columns=['IS_DELAYED'])
    y = df['IS_DELAYED']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_trainer = ModelTrainer(configs=MODEL_CONFIG)

    model = model_trainer.run_model_selection(
        X_train, y_train, X_test, y_test, n_iter=1
    )

    model, best_params, metrics = model_trainer.run_fine_tuning(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

    output_path = f'models/model_{model.lower()}.pkl'
    joblib.dump(model, output_path)