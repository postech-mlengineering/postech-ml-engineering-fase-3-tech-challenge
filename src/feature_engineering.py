import logging

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

    def _create_time_base(self):
        '''Cria colunas de tempo básicas e ordena os dados.'''
        self.df['SCHEDULED_DEPARTURE_DATETIME'] = pd.to_datetime(
            self.df[['YEAR', 'MONTH', 'DAY']].assign(
                hour=self.df['SCHEDULED_DEPARTURE'] // 100, 
                minute=self.df['SCHEDULED_DEPARTURE'] % 100
            )
        )
        self.df = self.df.sort_values(by=['TAIL_NUMBER', 'SCHEDULED_DEPARTURE_DATETIME'])

    def _apply_airplane_delayed(self):
        '''Lógica de atraso acumulado por aeronave.'''
        #atraso no voo anterior
        self.df['AIRPLANE_WAS_DELAYED'] = self.df.groupby('TAIL_NUMBER')['IS_DELAYED'].shift(1).fillna(0).astype(int)
        
        #janela de recuperação
        self.df['AIRPLANE_TIME_BETWEEN_FLIGHTS'] = self.df.groupby('TAIL_NUMBER')['SCHEDULED_DEPARTURE_DATETIME'].diff().dt.total_seconds() / 3600
        self.df['HAS_RECOVERY_WINDOW'] = (self.df['AIRPLANE_TIME_BETWEEN_FLIGHTS'] > 6).astype(int)
        
        #só considera atrasado se não teve tempo de recuperar
        self.df['AIRPLANE_WAS_DELAYED'] = ((self.df['AIRPLANE_WAS_DELAYED'] == 1) & (self.df['HAS_RECOVERY_WINDOW'] == 0)).astype(int)

    def _apply_momentum_features(self):
        '''Calcula o momentum de atrasos nos aeroportos (última 1 hora).'''
        #momentum origem
        self.df = self.df.sort_values(['ORIGIN_AIRPORT', 'SCHEDULED_DEPARTURE_DATETIME']).reset_index(drop=True)
        self.df['ORIGIN_AIRPORT_DELAY_MOMENTUM'] = (
            self.df.groupby('ORIGIN_AIRPORT')
            .rolling('1h', on='SCHEDULED_DEPARTURE_DATETIME', closed='left')['IS_DELAYED']
            .mean().reset_index(level=0, drop=True).fillna(0).values
        )

        #momentum destino
        self.df = self.df.sort_values(['DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE_DATETIME']).reset_index(drop=True)
        self.df['DESTINATION_AIRPORT_DELAY_MOMENTUM'] = (
            self.df.groupby('DESTINATION_AIRPORT')
            .rolling('1h', on='SCHEDULED_DEPARTURE_DATETIME', closed='left')['IS_DELAYED']
            .mean().reset_index(level=0, drop=True).fillna(0).values
        )
        
        #voos simultâneos na janela
        self.df = self.df.sort_values(by=['ORIGIN_AIRPORT', 'SCHEDULED_DEPARTURE_DATETIME'])
        self.df['ORIGIN_AIRPORT_FLIGHTS_ON_THE_SAME_WINDOW'] = (
            self.df.groupby('ORIGIN_AIRPORT')
            .rolling('1h', on='SCHEDULED_DEPARTURE_DATETIME')['DAY']
            .count().reset_index(level=0, drop=True).values - 1
        ).astype(int)

    def _apply_clustering(self):
        '''Cria perfis usando KMeans para Aeroportos, Rotas e Companhias.'''
        
        #aeroporto de origem
        profile_origem = self.df.groupby('ORIGIN_AIRPORT').agg({'IS_DELAYED': ['mean', 'count']}).reset_index()
        profile_origem.columns = ['ORIGIN_AIRPORT', 'ORIGIN_AIRPORT_DELAY_RATE', 'FLIGHT_VOLUME']
        profile_origem['ORIGIN_AIRPORT_PROFILE'] = self._run_clustering(profile_origem[['ORIGIN_AIRPORT_DELAY_RATE', 'FLIGHT_VOLUME']])
        
        #rota
        self.df['ROUTE'] = self.df['ORIGIN_AIRPORT'] + '_' + self.df['DESTINATION_AIRPORT']
        profile_route = self.df.groupby('ROUTE').agg({'IS_DELAYED': ['mean', 'count']}).reset_index()
        profile_route.columns = ['ROUTE', 'ROUTE_DELAY_RATE', 'FLIGHT_VOLUME']
        profile_route['ROUTE_PROFILE'] = self._run_clustering(profile_route[['ROUTE_DELAY_RATE', 'FLIGHT_VOLUME']])

        #airline
        profile_airline = self.df.groupby('AIRLINE').agg({'IS_DELAYED': ['mean', 'count']}).reset_index()
        profile_airline.columns = ['AIRLINE', 'AIRLINE_DELAY_RATE', 'FLIGHT_VOLUME']
        profile_airline['AIRLINE_PROFILE'] = self._run_clustering(profile_airline[['AIRLINE_DELAY_RATE', 'FLIGHT_VOLUME']])

        #merge
        self.df = self.df.merge(profile_origem[['ORIGIN_AIRPORT', 'ORIGIN_AIRPORT_DELAY_RATE', 'ORIGIN_AIRPORT_PROFILE']], on='ORIGIN_AIRPORT', how='left')
        self.df = self.df.merge(profile_route[['ROUTE', 'ROUTE_DELAY_RATE', 'ROUTE_PROFILE']], on='ROUTE', how='left')
        self.df = self.df.merge(profile_airline[['AIRLINE', 'AIRLINE_DELAY_RATE', 'AIRLINE_PROFILE']], on='AIRLINE', how='left')

    def _run_clustering(self, data_to_cluster):
        scaled_data = StandardScaler().fit_transform(data_to_cluster)
        k = get_best_k(scaled_data)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        return kmeans.fit_predict(scaled_data)

    def _apply_temporal_and_haul(self):
        '''Variáveis de data, feriados e tipo de voo.'''
        #estações e período do dia
        seasons = {12: 'SUMMER', 1: 'SUMMER', 2: 'SUMMER', 3: 'AUTUMN', 4: 'AUTUMN', 5: 'AUTUMN',
                   6: 'WINTER', 7: 'WINTER', 8: 'WINTER', 9: 'SPRING', 10: 'SPRING', 11: 'SPRING'}
        self.df['SEASON'] = self.df['MONTH'].map(seasons)
        self.df['IS_WEEKEND'] = self.df['DAY_OF_WEEK'].isin([6, 7]).astype(int)
        
        #feriados
        dates = pd.to_datetime(self.df['MONTH'].astype(str) + '-' + self.df['DAY'].astype(str) + '-2015')
        self.df['IS_HOLIDAY'] = dates.dt.date.isin(self.us_holidays).astype(int)
        
        #haul e hora
        self.df['HAUL_TYPE'] = pd.qcut(self.df['DISTANCE'], q=3, labels=['SHORT', 'MEDIUM', 'LONG'])
        self.df['SCHEDULED_DEPARTURE_HOUR'] = self.df['SCHEDULED_DEPARTURE'] // 100

    def _balance_data(self) -> pd.DataFrame:
        '''Realiza o undersampling, escalonamento e encoding final.'''
        cols = [
            'IS_DELAYED', 'AIRPLANE_WAS_DELAYED', 'ORIGIN_AIRPORT_DELAY_MOMENTUM',
            'DESTINATION_AIRPORT_DELAY_MOMENTUM', 'ORIGIN_AIRPORT_FLIGHTS_ON_THE_SAME_WINDOW',
            'HAUL_TYPE', 'MONTH', 'SCHEDULED_DEPARTURE_HOUR', 'IS_WEEKEND', 'IS_HOLIDAY'
        ]
        df = self.df[cols].copy()

        df_atrasados = df[df['IS_DELAYED'] == 1]
        df_pontuais = df[df['IS_DELAYED'] == 0]
        df_pontuais_bal = df_pontuais.sample(n=len(df_atrasados), random_state=42)
        
        df_balanced = pd.concat([df_atrasados, df_pontuais_bal]).sample(frac=1, random_state=42).reset_index(drop=True)

        #escalonamento
        cols_numericas = [
            'AIRPLANE_WAS_DELAYED', 'ORIGIN_AIRPORT_DELAY_MOMENTUM', 'DESTINATION_AIRPORT_DELAY_MOMENTUM',
            'ORIGIN_AIRPORT_FLIGHTS_ON_THE_SAME_WINDOW', 'MONTH', 'SCHEDULED_DEPARTURE_HOUR', 
            'IS_WEEKEND', 'IS_HOLIDAY'
        ]
        df_balanced[cols_numericas] = self.scaler.fit_transform(df_balanced[cols_numericas])

        #encoding
        df_balanced = pd.get_dummies(df_balanced, columns=['HAUL_TYPE'], dtype=int)
        
        return df_balanced

    def run_pipeline(self):
        '''Executa todo o processo de engenharia de dados.'''
        logging.info('Iniciando Feature Engineering')
        self.load_data()
        self._create_time_base()
        self._apply_airplane_delayed()
        self._apply_momentum_features()
        self._apply_clustering()
        self._apply_temporal_and_haul()
        
        df_balanced = self._balance_data()

        logging.info(f'Salvando dados processados em: {self.output_file_path}')
        df_balanced.to_pickle(self.output_file_path)
        return df_balanced


if __name__ == '__main__':
    feature_engineer = FeatureEngineer(input_file_path='data/curated/data.pkl')
    feature_engineer.run_pipeline(output_file_path='data/curated/features.pkl')