# Repositório para o Tech Challenge da Fase 3 da Pós-Graduação em Machine Learning Engineering da FIAP

Este repositório consiste em um pipeline completo de Machine Learning para **predição de atrasos em voos domésticos dos EUA**. A solução abrange desde a limpeza e engenharia de features até a seleção, otimização e persistência de modelos supervisionados e não supervisionados.

### Modelagem

**Modelos Não Supervisionados**

KMeans — Feature Engineering (Pré-processamento)

Aplicado durante a engenharia de features para gerar perfis de risco de aeroportos de origem, rotas e companhias aéreas. O número ideal de clusters (K) é determinado automaticamente pelo **método do cotovelo** via `KneeLocator`, garantindo que os perfis reflitam agrupamentos naturais nos dados. As variáveis geradas (`ORIGIN_AIRPORT_PROFILE`, `ROUTE_PROFILE`, `AIRLINE_PROFILE`) são utilizadas diretamente como features de entrada para o modelo supervisionado.

**Modelos Supervisionados**

A seleção do melhor modelo é feita via `RandomizedSearchCV` (CV=3), comparando o F1-Score ponderado entre os três candidatos abaixo. O vencedor passa por um fine-tuning adicional (CV=5, n_iter=10) antes de ser persistido.

XGBoost

Algoritmo de gradient boosting baseado em árvores de decisão com otimizações de velocidade e memória (`tree_method='hist'`). Combina diversas árvores fracas de forma sequencial, corrigindo os erros das anteriores por gradiente descendente. Destacou-se pela capacidade de capturar interações não lineares entre as features de atraso (momentum, perfis de aeroporto e histórico da aeronave).

Melhores hiperparâmetros encontrados:

| Parâmetro | Valor | Descrição |
|---|---|---|
| `n_estimators` | 500 | Número de árvores no ensemble |
| `max_depth` | 10 | Profundidade máxima de cada árvore |
| `learning_rate` | 0.05 | Taxa de aprendizado (passo do gradiente) |
| `subsample` | 1.0 | Fração de amostras usada por árvore |
| `colsample_bytree` | 0.6 | Fração de features amostradas por árvore |
| `gamma` | 0.1 | Ganho mínimo para realizar uma divisão |
| `min_child_weight` | 5 | Peso mínimo nas folhas (controla overfitting) |

Resultados no conjunto de teste:

| Métrica | Valor |
|---|---|
| Acurácia | 0.7484 |
| F1-Score (ponderado) | 0.7460 |
| Precisão (ponderada) | 0.7581 |
| Recall (ponderado) | 0.7484 |
| Melhor Score CV | 0.7463 |

Random Forest

Ensemble de árvores de decisão treinadas de forma independente com amostragem aleatória de dados e features (bagging). Robusto a overfitting e eficaz em dados com alta dimensionalidade, mas computacionalmente mais custoso em datasets grandes.

Gradient Boosting (scikit-learn)

Implementação clássica de gradient boosting sequencial do scikit-learn. Constrói árvores de forma aditiva, onde cada nova árvore aprende com os resíduos da anterior. Mais lento que o XGBoost para grandes volumes de dados, porém com controle preciso de regularização.

### Pré-requisitos

Certifique-se de ter o **Python 3.11+** instalado. Recomendamos o uso do **[uv](https://github.com/astral-sh/uv)**, um gerenciador de pacotes Python extremamente rápido escrito em Rust.

Se você ainda não tem o `uv` instalado:
```bash
pip install uv
```

### Instalação

Clone o repositório e configure o ambiente virtual:

```bash
git clone https://github.com/postech-mlengineering/postech-ml-engineering-fase-3-tech-challenge.git

cd postech-ml-engineering-fase-3-tech-challenge
```

> Os dados brutos (`flights.csv`, `airlines.csv`, `airports.csv`) devem estar em `data/raw/` antes de executar o pipeline.

---

### Como Executar

1. Criar o ambiente virtual:

```bash
uv venv
```

2. Ativar o ambiente virtual:

**Windows (PowerShell/CMD):**
```bash
.\.venv\Scripts\activate
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

3. Instalar as dependências:

```bash
uv pip install -r requirements.txt
```

4. Pipeline de ML:

Execute o pipeline completo de treinamento.

1. **Limpeza dos dados** — leitura dos CSVs brutos, cruzamento das tabelas, tratamento de nulos e remoção de voos cancelados/desviados
2. **Engenharia de Features** — criação de variáveis temporais, momentum de atrasos por aeroporto, perfis via KMeans e balanceamento por undersampling
3. **Separação de dados de treino/teste** — estratificada, proporção 80/20
4. **Seleção de modelo** — comparação com `RandomizedSearchCV` entre XGBoost, Random Forest e Gradient Boosting
5. **Fine-tuning** — otimização de hiperparâmetros do melhor modelo
6. **Persistência** — modelo salvo em `models/model_{nome}.pkl`

```bash
python main.py
```

### Dashboard

O projeto inclui um dashboard interativo em **Streamlit** para análise e inferência em tempo real.

```bash
cd frontend
streamlit run app.py
```

O dashboard conta com quatro páginas:

| Página | Descrição |
|---|---|
| **Sobre** | Documentação técnica do projeto: metodologia, pipeline e resultados |
| **Estatísticas** | Análise exploratória dos dados históricos e perfis de risco via clustering |
| **Performance** | Métricas do modelo: relatório de classificação, matriz de confusão, curva ROC e importância das features |
| **Predição de Atraso** | Interface para inferência em tempo real com gauge de probabilidade de atraso |


### Notebooks

Os notebooks com as análises exploratórias estão em `notebooks/`. Para executá-los no VS Code, selecione o kernel do `.venv` criado acima. Alternativamente, via terminal:

```bash
jupyter notebook
```

### Tecnologias

| Componente | Tecnologia | Versão | Descrição |
|---|---|---|---|
| Linguagem | Python | `>=3.11` | Linguagem para desenvolvimento de scripts |
| ML | Scikit-learn | `1.8.0` | Biblioteca para desenvolvimento de modelos de ML |
| ML | XGBoost | `3.2.0` | Biblioteca para desenvolvimento de modelos de ML |
| Análise de Dados | Pandas | `3.0.2` | Biblioteca para manipulação de dados |
| Análise de Dados | NumPy | `2.4.4` | Operações numéricas |
| Visualização | Plotly | `6.7.0` | Biblioteca para criação de gráficos dinâmicos e interativos |
| Visualização | Streamlit | `1.57.0` | Framework para desenvolvimento de aplicativo web |
| Serialização | Joblib | `1.5.3` | Ferramenta para persistência de modelos de ML e execução de tarefas |
| Feriados | Holidays | `0.97` | Biblioteca para identificação de feriados dos EUA |
| Kneedle | Kneed | `0.8.6` | Biblioteca para identificação automática do k ideal no KMeans |
| Gerenciamento | uv | — | Gerenciador de ambientes virtuais para isolamento de dependências |


## Estrutura do Projeto

```
.
├── data/
│   ├── raw/                    # Dados brutos (flights, airlines, airports)
│   └── curated/                # Dados processados pelo pipeline
├── models/                     # Artefatos salvos dos modelos
├── notebooks/                  # Análises exploratórias
├── frontend/                   # Dashboard Streamlit
│   ├── app.py                  # Ponto de entrada e navegação
│   ├── pages/                  # Páginas: about, analytics, performance, prediction
│   ├── charts/                 # Gráficos Plotly com tema Synthwave
│   └── services/               # Carregamento de dados, modelo e clustering
├── src/
│   ├── data_cleaning.py        # Limpeza e cruzamento dos dados brutos
│   ├── feature_engineering.py  # Engenharia de features e balanceamento
│   ├── model_config.py         # Configuração dos modelos e hiperparâmetros
│   ├── model_trainer.py        # Seleção de modelo e fine-tuning
│   └── utils/
│       ├── helpers.py          # Funções auxiliares (elbow method, etc.)
│       ├── plots.py
│       └── evaluation_plots.py
├── main.py                     # Ponto de entrada do pipeline de ML
└── requirements.txt
```

---

## Colaboradores

[Jorge Platero](https://github.com/jorgeplatero)

[Leandro Delisposti](https://github.com/LeandroDelisposti)

[Hugo Rodrigues](https://github.com/Nokard)
