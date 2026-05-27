import warnings
import logging
import joblib
import os 

from sklearn.model_selection import train_test_split

from src.data_cleaning import DataCleaner   
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer


warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main() -> None:
    '''
    Executa o pipeline completo.

    O pipeline consiste em cinco etapas principais:

        1. Limpeza dos dados: Processamento dos dados brutos.
        2. Engenharia de Features: Processamento dos dados tratados para o modelo.
        3. Separando dos dados: Divisão dos dados em conjuntos de treino e teste.
        4. Seleção de Modelo: Comparação de diferentes algoritmos.
        5. Fine-tuning: Otimização de hiperparâmetros do melhor modelo.
        6. Persistência: Salvamento do modelo e metadados em disco.

    Raises:
        Exception: Erro durante a execução do pipeline.
    '''
    try:
        os.makedirs('models', exist_ok=True)

        logger.info('ETAPA 1: Iniciando limpeza de dados')
        data_cleaner = DataCleaner(
            input_folder_path='data/raw', 
            output_file_path='data/curated/data.pkl'
        )
        data_cleaner.run_data_cleaning()

        logger.info('ETAPA 2: Iniciando engenharia de features')
        feature_engineer = FeatureEngineer(
            input_file_path='data/curated/data.pkl', 
            output_file_path='data/curated/features.pkl'
        )
        df = feature_engineer.run_pipeline()

        logger.info('ETAPA 3: Separando dados de treino e teste')
        X = df.drop(columns=['IS_DELAYED'])
        y = df['IS_DELAYED']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info('ETAPA 4: Iniciando seleção do modelo')
        trainer = ModelTrainer()
        best_model_name = trainer.run_model_selection(
            X_train, y_train, X_test, y_test, 
            n_iter=1
        )

        logger.info(f'ETAPA 5: Iniciando fine-tuning para o modelo: {best_model_name}')
        model, best_params, metrics = trainer.run_fine_tuning(
            model_name=best_model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_iter=10
        )

        model_path = f'/models/model_{best_model_name.lower()}.pkl'
        logger.info(f'ETAPA 6: Salvando modelo em {model_path}')
        
        joblib.dump(model, model_path)
        
        logger.info('Pipeline executado com sucesso')

    except Exception as e:
        logger.error(f'Erro durante a execução do pipeline: {e}', exc_info=True)


if __name__ == '__main__':
    main()