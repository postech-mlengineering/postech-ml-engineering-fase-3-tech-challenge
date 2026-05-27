import logging
import joblib

import holidays
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils.helpers import get_best_k


logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, input_file_path: str, output_file_path: str):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.df = None
        self.us_holidays = holidays.US(years=2015)
        self.scaler = StandardScaler()

    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_pickle(self.input_file_path)
        return self.df

    def _create_datetime(self):
        self.df['SCHEDULED_DEPARTURE_DATETIME'] = pd.to_datetime(
            self.df[['YEAR', 'MONTH', 'DAY']].assign(
                hour=self.df['SCHEDULED_DEPARTURE'] // 100, 
                minute=self.df['SCHEDULED_DEPARTURE'] % 100
            )
        )
        self.df = self.df.sort_values(by=['TAIL_NUMBER', 'SCHEDULED_DEPARTURE_DATETIME'])

    def _apply_airplane_delayed(self):
        self.df['AIRPLANE_WAS_DELAYED'] = self.df.groupby('TAIL_NUMBER')['IS_DELAYED'].shift(1).fillna(0).astype(int)
        self.df['AIRPLANE_TIME_BETWEEN_FLIGHTS'] = self.df.groupby('TAIL_NUMBER')['SCHEDULED_DEPARTURE_DATETIME'].diff().dt.total_seconds() / 3600
        self.df['HAS_RECOVERY_WINDOW'] = (self.df['AIRPLANE_TIME_BETWEEN_FLIGHTS'] > 6).astype(int)
        self.df['AIRPLANE_WAS_DELAYED'] = ((self.df['AIRPLANE_WAS_DELAYED'] == 1) & (self.df['HAS_RECOVERY_WINDOW'] == 0)).astype(int)

    def _apply_momentum_features(self):
        # Momentum origem
        self.df = self.df.sort_values(['ORIGIN_AIRPORT', 'SCHEDULED_DEPARTURE_DATETIME']).reset_index(drop=True)
        self.df['ORIGIN_AIRPORT_DELAY_MOMENTUM'] = (
            self.df.groupby('ORIGIN_AIRPORT')
            .rolling('1h', on='SCHEDULED_DEPARTURE_DATETIME', closed='left')['IS_DELAYED']
            .mean().reset_index(level=0, drop=True).fillna(0).values
        )
        # Momentum destino
        self.df = self.df.sort_values(['DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE_DATETIME']).reset_index(drop=True)
        self.df['DESTINATION_AIRPORT_DELAY_MOMENTUM'] = (
            self.df.groupby('DESTINATION_AIRPORT')
            .rolling('1h', on='SCHEDULED_DEPARTURE_DATETIME', closed='left')['IS_DELAYED']
            .mean().reset_index(level=0, drop=True).fillna(0).values
        )
        # Simultâneos
        self.df = self.df.sort_values(by=['ORIGIN_AIRPORT', 'SCHEDULED_DEPARTURE_DATETIME'])
        self.df['ORIGIN_AIRPORT_FLIGHTS_ON_THE_SAME_WINDOW'] = (
            self.df.groupby('ORIGIN_AIRPORT')
            .rolling('1h', on='SCHEDULED_DEPARTURE_DATETIME')['DAY']
            .count().reset_index(level=0, drop=True).values - 1
        ).astype(int)

    def _apply_clustering(self):
        for col in ['ORIGIN_AIRPORT', 'AIRLINE']:
            prof = self.df.groupby(col).agg({'IS_DELAYED': ['mean', 'count']}).reset_index()
            prof.columns = [col, f'{col}_DELAY_RATE', 'VOL']
            prof[f'{col}_PROFILE'] = self._run_clustering(prof[[f'{col}_DELAY_RATE', 'VOL']])
            self.df = self.df.merge(prof[[col, f'{col}_DELAY_RATE', f'{col}_PROFILE']], on=col, how='left')

    def _run_clustering(self, data_to_cluster):
        scaled_data = StandardScaler().fit_transform(data_to_cluster)
        k = get_best_k(scaled_data)
        return KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled_data)

    def _create_contextual_features(self):
        seasons = {
            12: 'WINTER', 1: 'WINTER', 2: 'WINTER', 3: 'SPRING', 4: 'SPRING', 5: 'SPRING',
            6: 'SUMMER', 7: 'SUMMER', 8: 'SUMMER', 9: 'AUTUMN', 10: 'AUTUMN', 11: 'AUTUMN'
        }
        self.df['SEASON'] = self.df['MONTH'].map(seasons)
        self.df['IS_WEEKEND'] = self.df['DAY_OF_WEEK'].isin([6, 7]).astype(int)
        dates = pd.to_datetime(self.df['MONTH'].astype(str) + '-' + self.df['DAY'].astype(str) + '-2015')
        self.df['IS_HOLIDAY'] = dates.dt.date.isin(self.us_holidays).astype(int)
        self.df['HAUL_TYPE'] = pd.qcut(self.df['DISTANCE'], q=3, labels=['SHORT', 'MEDIUM', 'LONG'])
        self.df['SCHEDULED_DEPARTURE_HOUR'] = self.df['SCHEDULED_DEPARTURE'] // 100

    def _select_features(self):
        '''Filtra apenas as colunas que serão utilizadas pelo modelo.'''
        cols = [
            'IS_DELAYED', 'AIRPLANE_WAS_DELAYED', 'ORIGIN_AIRPORT_DELAY_MOMENTUM',
            'DESTINATION_AIRPORT_DELAY_MOMENTUM', 'ORIGIN_AIRPORT_FLIGHTS_ON_THE_SAME_WINDOW',
            'HAUL_TYPE', 'MONTH', 'SCHEDULED_DEPARTURE_HOUR', 'IS_WEEKEND', 'IS_HOLIDAY'
        ]
        self.df = self.df[cols].copy()

    def _encode_features(self):
        '''Transforma variáveis categóricas em numéricas.'''
        self.df = pd.get_dummies(self.df, columns=['HAUL_TYPE'], dtype=int)

    def _balance_data(self):
        '''Realiza o undersampling para lidar com classes desbalanceadas.'''
        df_atrasados = self.df[self.df['IS_DELAYED'] == 1]
        df_pontuais = self.df[self.df['IS_DELAYED'] == 0]
        df_pontuais_bal = df_pontuais.sample(n=len(df_atrasados), random_state=42)
        self.df = pd.concat([df_atrasados, df_pontuais_bal]).sample(frac=1, random_state=42).reset_index(drop=True)

    def _scale_features(self):
        '''Normaliza as features numéricas para a mesma escala.'''
        cols_to_scale = [
            'AIRPLANE_WAS_DELAYED', 'ORIGIN_AIRPORT_DELAY_MOMENTUM', 'DESTINATION_AIRPORT_DELAY_MOMENTUM',
            'ORIGIN_AIRPORT_FLIGHTS_ON_THE_SAME_WINDOW', 'MONTH', 'SCHEDULED_DEPARTURE_HOUR', 
            'IS_WEEKEND', 'IS_HOLIDAY'
        ]
        self.df[cols_to_scale] = self.scaler.fit_transform(self.df[cols_to_scale])

    def run_pipeline(self):
        logging.info('Iniciando Pipeline de Engenharia de Features')
        self.load_data()
        
        self._create_datetime()
        self._create_contextual_features()

        self._apply_airplane_delayed()
        self._apply_momentum_features()
        self._apply_clustering()

        self._select_features()
        self._encode_features()
        self._balance_data()
        self._scale_features()

        logging.info(f'Salvando dados processados em: {self.output_file_path}')
        self.df.to_pickle(self.output_file_path)
        joblib.dump(self.scaler, 'models/scaler.pkl') 
        return self.df, self.scaler

if __name__ == '__main__':
    feature_engineer = FeatureEngineer(
        input_file_path='../data/curated/data.pkl', 
        output_file_path='../data/curated/features.pkl'
    )
    feature_engineer.run_pipeline()