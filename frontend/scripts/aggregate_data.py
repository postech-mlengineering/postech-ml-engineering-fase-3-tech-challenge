import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

def main():
    print("Iniciando pré-processamento e agregação de dados...")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    print(f"Lendo dados de {data_dir}...")
    # Carregar os dados
    df_flights = pd.read_csv(os.path.join(data_dir, 'flights.csv'), low_memory=False)
    df_airlines = pd.read_csv(os.path.join(data_dir, 'airlines.csv'))
    df_airports = pd.read_csv(os.path.join(data_dir, 'airports.csv'))

    # Renomeando colunas para merge
    df_airlines = df_airlines.rename(columns={'IATA_CODE': 'AIRLINE_IATA', 'AIRLINE': 'AIRLINE_NAME'})
    
    df_airport_origin = df_airports.rename(columns={
        'IATA_CODE': 'ORIGIN_AIRPORT', 'AIRPORT': 'ORIGIN_AIRPORT_NAME', 'CITY': 'ORIGIN_CITY',
        'STATE': 'ORIGIN_STATE', 'COUNTRY': 'ORIGIN_COUNTRY', 'LATITUDE': 'ORIGIN_LATITUDE', 'LONGITUDE': 'ORIGIN_LONGITUDE'
    })
    
    df_airport_dest = df_airports.rename(columns={
        'IATA_CODE': 'DESTINATION_AIRPORT', 'AIRPORT': 'DEST_AIRPORT_NAME', 'CITY': 'DEST_CITY',
        'STATE': 'DEST_STATE', 'COUNTRY': 'DEST_COUNTRY', 'LATITUDE': 'DEST_LATITUDE', 'LONGITUDE': 'DEST_LONGITUDE'
    })

    print("Realizando joins...")
    # Merge
    df = df_flights.merge(df_airlines, left_on='AIRLINE', right_on='AIRLINE_IATA', how='left')
    df = df.merge(df_airport_origin, on='ORIGIN_AIRPORT', how='left')
    df = df.merge(df_airport_dest, on='DESTINATION_AIRPORT', how='left')

    print("Tratando nulos e filtrando dados...")
    # Tratamento de Nulos
    delay_cols = ['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']
    df[delay_cols] = df[delay_cols].fillna(0)
    
    # Remover colunas desnecessárias
    if 'CANCELLATION_REASON' in df.columns:
        df = df.drop(columns=['CANCELLATION_REASON'])
        
    df = df.dropna(subset=['ARRIVAL_DELAY', 'ORIGIN_CITY', 'DEST_CITY', 'ORIGIN_LATITUDE', 'DEST_LATITUDE', 'TAIL_NUMBER'])

    # Removendo voos cancelados e desviados
    df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)]
    
    # Criação do target para modelo supervisionado (1 se atraso > 15 min, 0 caso contrário)
    df['IS_DELAYED'] = (df['ARRIVAL_DELAY'] > 15).astype(int)

    # 1. Agregação para Análise Geográfica (Rotas)
    print("Gerando agregação de rotas...")
    route_delays = df.groupby(
        ['ORIGIN_AIRPORT', 'ORIGIN_AIRPORT_NAME', 'ORIGIN_CITY', 'ORIGIN_STATE', 'ORIGIN_LATITUDE', 'ORIGIN_LONGITUDE',
         'DESTINATION_AIRPORT', 'DEST_AIRPORT_NAME', 'DEST_CITY', 'DEST_STATE', 'DEST_LATITUDE', 'DEST_LONGITUDE',
         'AIRLINE_NAME']
    ).agg(
        AVG_ARRIVAL_DELAY=('ARRIVAL_DELAY', 'mean'),
        AVG_DEPARTURE_DELAY=('DEPARTURE_DELAY', 'mean'),
        FLIGHT_COUNT=('FLIGHT_NUMBER', 'count')
    ).reset_index()
    
    route_delays.to_parquet(os.path.join(data_dir, 'route_delays.parquet'), index=False)

    # 2. Agregação para Mapa de Calor (Aeroportos de Origem)
    print("Gerando agregação de aeroportos (Heatmap)...")
    airport_delays = df.groupby(
        ['ORIGIN_AIRPORT', 'ORIGIN_AIRPORT_NAME', 'ORIGIN_CITY', 'ORIGIN_STATE', 'ORIGIN_LATITUDE', 'ORIGIN_LONGITUDE']
    ).agg(
        AVG_DEPARTURE_DELAY=('DEPARTURE_DELAY', 'mean'),
        FLIGHT_COUNT=('FLIGHT_NUMBER', 'count'),
        TOTAL_DELAYED=('IS_DELAYED', 'sum')
    ).reset_index()
    
    # Calcular percentual de voos atrasados
    airport_delays['DELAY_PERCENTAGE'] = (airport_delays['TOTAL_DELAYED'] / airport_delays['FLIGHT_COUNT']) * 100
    airport_delays.to_parquet(os.path.join(data_dir, 'airport_delays.parquet'), index=False)

    # 3. Agregação para Modelo Não Supervisionado (Clusterização de Companhias)
    print("Gerando estatísticas por companhia aérea...")
    airline_stats = df.groupby('AIRLINE_NAME').agg(
        TAXI_OUT=('TAXI_OUT', 'mean'),
        TAXI_IN=('TAXI_IN', 'mean'),
        AIRLINE_DELAY=('AIRLINE_DELAY', 'mean'),
        WEATHER_DELAY=('WEATHER_DELAY', 'mean'),
        FLIGHT_VOLUME=('FLIGHT_NUMBER', 'count')
    ).reset_index()
    
    airline_stats.to_parquet(os.path.join(data_dir, 'airline_stats.parquet'), index=False)

    # 4. Treinando um modelo Supervisionado Leve para deploy
    print("Treinando modelo de classificação (Logistic Regression)...")
    # Para o dashboard, vamos usar um subconjunto de features fáceis de obter
    features = ['MONTH', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE', 'DISTANCE', 'AIRLINE_NAME']
    
    # Vamos pegar uma amostra de 500k linhas para treinar mais rápido
    df_sample = df.sample(n=500000, random_state=42)
    X = df_sample[features].copy()
    y = df_sample['IS_DELAYED']
    
    # Label Encoder para Airline Name
    le_airline = LabelEncoder()
    X['AIRLINE_NAME'] = le_airline.fit_transform(X['AIRLINE_NAME'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_scaled, y)
    
    # Salvando os artefatos do modelo
    joblib.dump(model, os.path.join(models_dir, 'logistic_regression_model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(le_airline, os.path.join(models_dir, 'le_airline.pkl'))
    
    print("Modelos e agregadores salvos com sucesso!")
    print("Fim do processo.")

if __name__ == '__main__':
    main()
