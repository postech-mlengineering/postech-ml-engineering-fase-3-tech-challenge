from typing import List
import logging

import pandas as pd


logger = logging.getLogger(__name__)


class DataCleaner:
    '''
    Classe responsável pelo carregamento, cruzamento e limpeza dos dados brutos.
    '''
    def __init__(self, input_folder_path: str, output_file_path: str):
        self.input_folder_path = input_folder_path
        self.output_file_path = output_file_path
        self.delay_cols: List[str] = [
            'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 
            'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'
        ]

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''Carrega os arquivos brutos.'''
        logger.info('Carregando arquivos brutos...')
        df_flights = pd.read_csv(f'{self.input_folder_path}/flights.csv')
        df_airlines = pd.read_csv(f'{self.input_folder_path}/airlines.csv')
        df_airports = pd.read_csv(f'{self.input_folder_path}/airports.csv')
        return df_flights, df_airlines, df_airports

    def _merge_datasets(self, df_flights: pd.DataFrame, df_airlines: pd.DataFrame, df_airports: pd.DataFrame) -> pd.DataFrame:
        '''Realiza os cruzamentos entre as tabelas.'''
        logger.info('Realizando merges dos datasets...')
        
        df = df_flights.merge(
            df_airlines.rename(columns={'AIRLINE': 'AIRLINE_NAME'}), 
            left_on='AIRLINE', right_on='IATA_CODE', how='left'
        ).drop(columns=['IATA_CODE'])

        df_airports_renamed = df_airports.rename(columns={'AIRPORT': 'AIRPORT_NAME'})

        df = df.merge(
            df_airports_renamed.add_prefix('ORIGIN_'), 
            left_on='ORIGIN_AIRPORT', right_on='ORIGIN_IATA_CODE', how='left'
        ).drop(columns=['ORIGIN_IATA_CODE'])

        df = df.merge(
            df_airports_renamed.add_prefix('DESTINATION_'),
            left_on='DESTINATION_AIRPORT', right_on='DESTINATION_IATA_CODE', how='left'
        ).drop(columns=['DESTINATION_IATA_CODE'])
        
        return df

    def run_data_cleaning(self) -> pd.DataFrame:
        '''
        Executa o fluxo completo de limpeza e salva o arquivo final em .pkl.
        '''
        df_flights, df_airlines, df_airports = self._load_data()
        df = self._merge_datasets(df_flights, df_airlines, df_airports)

        logger.info('Iniciando limpeza de nulos e filtros...')
        
        df[self.delay_cols] = df[self.delay_cols].fillna(0)
        df = df.drop(columns=['CANCELLATION_REASON'])

        subset_drop = [
            'ARRIVAL_DELAY', 'ORIGIN_CITY', 'DESTINATION_CITY', 
            'ORIGIN_LATITUDE', 'DESTINATION_LATITUDE', 'TAIL_NUMBER'
        ]
        df = df.dropna(subset=subset_drop)

        df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)]

        df['IS_DELAYED'] = (df['ARRIVAL_DELAY'] > 15).astype(int)

        output_file_path = self.output_file_path
        df.to_pickle(output_file_path)
        logger.info(f'Dados tratados salvos com sucesso em: {output_file_path}')

        return df
    

if __name__ == '__main__':
    input_folder_path = '../../data/raw'
    output_file_path = '../../data/curated/data.pkl'
    
    data_cleaner = DataCleaner(input_folder_path=input_folder_path, output_file_path=output_file_path)
    data_cleaner.run_data_cleaning()