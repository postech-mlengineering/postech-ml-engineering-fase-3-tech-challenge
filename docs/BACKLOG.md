### Modelo Supervisionado:
- [ ] **Feature Engineering no Modelo Supervisionado**: O grupo (cluster) da companhia aérea, identificado via K-Means, pode ser utilizado como uma excelente variável de entrada para o modelo preditivo de Machine Learning.
- [ ] **Isolamento de Outliers**: Podemos testar a remoção ou o tratamento diferenciado dos voos classificados como anomalias durante o treinamento do modelo supervisionado. Modelos costumam ter melhor performance preditiva quando não são influenciados por casos tão extremos.
- [ ] **Análise Profunda das Anomalias**: Investigar qualitativamente onde essas anomalias do Isolation Forest ocorreram (rotas específicas, épocas do ano, ou sob quais condições climáticas) para elaborar estratégias de mitigação precisas.

### Modelo Não Supervisionado:
- [ ] **Otimização dos Hiperparâmetros do K-Means**: Embora o método "Cotovelo" tenha sido usado para sugerir 4 clusters, o desempenho do Isolation Forest variou significativamente com diferentes valores. Seria valioso aplicar métodos mais robustos, como o Silhouette Score ou Gap Statistic, para validar o número ideal de clusters.
- [ ] **Explorar Algoritmos Alternativos de Agrupamento**: Avaliar outros algoritmos de clustering, como o DBSCAN (que detecta outliers de forma nativa) ou o Gaussian Mixture Models (GMM), para comparar a estrutura dos grupos encontrados.
- [ ] **Análise Multivariada dos Clusters**: Utilizar gráficos de dispersão com coloração por variável (em vez de apenas por cluster) ou gráficos de radar para identificar as características distintivas de cada grupo de anomalias.

### Infraestrutura e MLOps:

