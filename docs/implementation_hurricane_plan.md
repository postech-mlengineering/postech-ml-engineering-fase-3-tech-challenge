# Plano de Implementação – Análise de Atrasos de Voo com Furacões

Este documento descreve o plano detalhado para integrar dados de furacões à análise de atrasos de voos.

* **Objetivo**: Relacionar atrasos com ocorrências de furacões nos EUA.
* **Fonte de Dados**: NOAA HURDAT2 (arquivo CSV).
* **Variáveis Criadas**:
  * `hurricane_nearby` – flag booleana indicando presença de furacão próximo ao aeroporto (raio configurável).
  * `hurricane_category` – categoria do furacão (opcional).
* **Etapas Principais**:
  1. Download e limpeza dos dados de furacões.
  2. Função `haversine` para cálculo de distância.
  3. Geração da flag `hurricane_nearby` (janela temporal configurável).
  4. Análise exploratória (boxplot, linha temporal, teste de hipóteses).
  5. Atualização do pipeline supervisionado (incluindo a nova feature).
  6. Avaliação de métricas antes e depois da inclusão.

## Perguntas ao Usuário
- Janela temporal (ex.: ±1 dia, ±2 dias)?
- Raio de influência (ex.: 200 km)?
- Incluir furacões do Pacífico?
- Período de análise (ex.: 2015‑2024)?
- Local para armazenar o CSV de furacões (`data/external/` ou outro caminho)?

## Próximos Passos
1. Responder às perguntas acima.
2. Executar o notebook `modelo_supervisionado_hurricane.ipynb` que contém a implementação.

*Este plano será utilizado como referência para o desenvolvimento.*
