# Tech Challenge — Fase 3 · Pós-Graduação em Machine Learning Engineering · FIAP

Este repositório contém a solução desenvolvida para o Tech Challenge da Fase 3 da Pós-Graduação em Machine Learning Engineering da FIAP.

O projeto consiste em um pipeline completo de Machine Learning para **predição de atrasos em voos domésticos dos EUA**. A solução abrange desde a limpeza e engenharia de features até a seleção, otimização e persistência de modelos supervisionados e não supervisionados.

---

## Modelos Utilizados

### Modelos Não Supervisionados

#### KMeans — Feature Engineering (Pré-processamento)
Aplicado durante a engenharia de features para gerar perfis de risco de aeroportos de origem, rotas e companhias aéreas. O número ideal de clusters (K) é determinado automaticamente pelo **método do cotovelo** via `KneeLocator`, garantindo que os perfis reflitam agrupamentos naturais nos dados. As variáveis geradas (`ORIGIN_AIRPORT_PROFILE`, `ROUTE_PROFILE`, `AIRLINE_PROFILE`) são utilizadas diretamente como features de entrada para o modelo supervisionado.

---

### Modelos Supervisionados — Pipeline Principal

A seleção do melhor modelo é feita via `RandomizedSearchCV` (CV=3), comparando o F1-Score ponderado entre os três candidatos abaixo. O vencedor passa por um fine-tuning adicional (CV=5, n_iter=10) antes de ser persistido.

#### XGBoost ✅ Melhor Modelo
Algoritmo de gradient boosting baseado em árvores de decisão com otimizações de velocidade e memória (`tree_method='hist'`). Combina diversas árvores fracas de forma sequencial, corrigindo os erros das anteriores por gradiente descendente. Destacou-se pela capacidade de capturar interações não lineares entre as features de atraso (momentum, perfis de aeroporto e histórico da aeronave).

**Melhores hiperparâmetros encontrados:**

| Parâmetro | Valor | Descrição |
|---|---|---|
| `n_estimators` | 500 | Número de árvores no ensemble |
| `max_depth` | 10 | Profundidade máxima de cada árvore |
| `learning_rate` | 0.05 | Taxa de aprendizado (passo do gradiente) |
| `subsample` | 1.0 | Fração de amostras usada por árvore |
| `colsample_bytree` | 0.6 | Fração de features amostradas por árvore |
| `gamma` | 0.1 | Ganho mínimo para realizar uma divisão |
| `min_child_weight` | 5 | Peso mínimo nas folhas (controla overfitting) |

**Resultados no conjunto de teste:**

| Métrica | Valor |
|---|---|
| Acurácia | 0.7484 |
| F1-Score (ponderado) | 0.7460 |
| Precisão (ponderada) | 0.7581 |
| Recall (ponderado) | 0.7484 |
| Melhor Score CV | 0.7463 |

---

#### Random Forest
Ensemble de árvores de decisão treinadas de forma independente com amostragem aleatória de dados e features (bagging). Robusto a overfitting e eficaz em dados com alta dimensionalidade, mas computacionalmente mais custoso em datasets grandes.

#### Gradient Boosting (scikit-learn)
Implementação clássica de gradient boosting sequencial do scikit-learn. Constrói árvores de forma aditiva, onde cada nova árvore aprende com os resíduos da anterior. Mais lento que o XGBoost para grandes volumes de dados, porém com controle preciso de regularização.

---

## Pré-requisitos

Certifique-se de ter o **Python 3.11+** instalado. Recomendamos o uso do **[uv](https://github.com/astral-sh/uv)**, um gerenciador de pacotes Python extremamente rápido escrito em Rust.

Se você ainda não tem o `uv` instalado:
```bash
pip install uv
```

---

## Instalação

Clone o repositório e configure o ambiente virtual:

```bash
git clone https://github.com/postech-mlengineering/postech-ml-engineering-fase-3-tech-challenge-api.git
cd postech-ml-engineering-fase-3-tech-challenge-api
```

### 1. Criar o Ambiente Virtual

```bash
uv venv
```

### 2. Ativar o Ambiente Virtual

**Windows (PowerShell/CMD):**
```bash
.\.venv\Scripts\activate
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 3. Instalar as Dependências

```bash
uv pip install -r requirements.txt
```

> Os dados brutos (`flights.csv`, `airlines.csv`, `airports.csv`) devem estar em `data/raw/` antes de executar o pipeline.

---

## Como Executar

### Pipeline de Machine Learning

Execute o pipeline completo de treinamento. Ele percorre as seguintes etapas automaticamente:

1. **Limpeza dos dados** — leitura dos CSVs brutos, cruzamento das tabelas, tratamento de nulos e remoção de voos cancelados/desviados
2. **Engenharia de Features** — criação de variáveis temporais, momentum de atrasos por aeroporto, perfis via KMeans e balanceamento por undersampling
3. **Divisão treino/teste** — estratificada, proporção 80/20
4. **Seleção de modelo** — comparação com `RandomizedSearchCV` (CV=3) entre XGBoost, Random Forest e Gradient Boosting
5. **Fine-tuning** — otimização de hiperparâmetros do melhor modelo (CV=5, n_iter=10)
6. **Persistência** — modelo salvo em `models/best_model_{nome}.pkl`

```bash
python main.py
```

---

### (Opcional) Notebooks de Análise

Os notebooks com as análises exploratórias estão em `notebooks/`. Para executá-los no VS Code, selecione o kernel do `.venv` criado acima. Alternativamente, via terminal:

```bash
jupyter notebook
```

---

## Tecnologias

| Componente | Tecnologia | Versão | Descrição |
|---|---|---|---|
| Linguagem | Python | >=3.11 | Linguagem de desenvolvimento |
| ML | Scikit-learn | 1.8.0 | Modelos, pré-processamento e avaliação |
| ML | XGBoost | 3.2.0 | Classificador baseado em gradient boosting otimizado |
| Análise de Dados | Pandas | 3.0.2 | Manipulação e transformação de dados |
| Análise de Dados | NumPy | 2.4.4 | Operações numéricas |
| Visualização | Matplotlib / Seaborn | 3.10.9 / 0.13.2 | Gráficos de avaliação dos modelos |
| Persistência | Joblib | 1.5.3 | Serialização de modelos |
| Feriados | Holidays | — | Features de feriados dos EUA |
| Elbow Method | Kneed | — | Identificação automática do K ideal no KMeans |
| Gerenciamento | uv | — | Gerenciador de ambientes e pacotes Python |

---

## Estrutura do Projeto

```
.
├── data/
│   ├── raw/                    # Dados brutos (flights, airlines, airports)
│   └── curated/                # Dados processados pelo pipeline
├── models/                     # Artefatos salvos dos modelos
├── notebooks/                  # Análises exploratórias
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

Hugo Rodrigues
