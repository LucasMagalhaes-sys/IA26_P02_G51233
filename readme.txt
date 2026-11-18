========================================================================
PROJETO 02 - INTELIGÊNCIA ARTIFICIAL (2026)
GRUPO: G51233
========================================================================

LINK DO REPOSITÓRIO GITHUB:
https://github.com/LucasMagalhaes-sys/IA26_P02_G51233

------------------------------------------------------------------------
IDENTIFICAÇÃO DA EQUIPE:
------------------------------------------------------------------------
- Aluno 1: 51233 - Lucas Lima Magalhães

------------------------------------------------------------------------
DESCRIÇÃO GERAL DO PROJETO:
------------------------------------------------------------------------
Este projeto aborda a aplicação integral de técnicas fundamentais de
Ciência de Dados e Inteligência Artificial sobre dois conjuntos de dados
públicos do Kaggle:
1. California Housing Prices (Habitação e Mercado Imobiliário)
2. Home Credit / Credit Risk Dataset (Risco de Crédito e Incumprimento)

Em conformidade estrita com as exigências pedagógicas, os algoritmos
principais de Machine Learning (Árvore de Decisão, K-Means e Regras de
Associação Apriori), bem como as rotinas de pré-processamento,
escalonamento, particionamento estratificado e métricas de avaliação foram
implementados 100% FROM SCRATCH em Python, NumPy e Pandas puro, sem
nenhum recurso a bibliotecas de aprendizado de máquina externas (como sklearn).

------------------------------------------------------------------------
ESTRUTURA DE PASTAS E ARQUIVOS:
------------------------------------------------------------------------
IA26_P02_G51233/
|-- data/
|   |-- housing.csv             (Base completa do California Housing Prices)
|   |-- home_credit.csv         (Base completa de Risco de Crédito - 32k linhas)
|   |-- home_credit_sample.csv  (Amostra estratificada de 5.000 linhas, seed=42)
|
|-- src/
|   |-- __init__.py             (Inicialização do pacote local)
|   |-- download_data.py        (Script de download automatizado via kagglehub)
|   |-- preprocessing.py        (StandardScaler, MinMaxScaler, LabelEncoder, impute, split)
|   |-- metrics.py              (Matriz de confusão, acurácia, precisão, recall, F1, WCSS, silhueta)
|   |-- decision_tree.py        (Árvore de Decisão from scratch vetorizada)
|   |-- kmeans.py               (K-Means from scratch com k-means++ e histórico de inércia)
|   |-- apriori.py              (Algoritmo Apriori e regras de associação from scratch)
|
|-- tests/
|   |-- test_core.py            (Bateria de 12 testes unitários automatizados)
|
|-- 01_eda.ipynb                (Notebook 01: Análise Exploratória de Dados executado)
|-- 02_classify.ipynb           (Notebook 02: Classificação por Árvore de Decisão executado)
|-- 03_clusters.ipynb           (Notebook 03: Agrupamento K-Means executado)
|-- 04_assoc_rules.ipynb        (Notebook 04: Regras de Associação Apriori executado)
|
|-- requirements.txt            (Lista de dependências necessárias)
|-- readme.txt                  (Este arquivo de documentação e guia do projeto)
|-- IA_P02_G51233.zip           (Pacote compactado final de entrega)

------------------------------------------------------------------------
ORIGEM DOS DATASETS:
------------------------------------------------------------------------
- Housing:
  Nome: California Housing Prices
  Slug Kaggle: camnugent/california-housing-prices
  URL: https://www.kaggle.com/datasets/camnugent/california-housing-prices

- Home Credit:
  Nome: Home Credit / Credit Risk Dataset
  Slug Kaggle: laotse/credit-risk-dataset
  URL: https://www.kaggle.com/datasets/laotse/credit-risk-dataset

------------------------------------------------------------------------
INSTRUÇÕES DE INSTALAÇÃO E EXECUÇÃO:
------------------------------------------------------------------------
1. Criação e ativação de ambiente virtual Python (versão recomendada >= 3.10):
   python -m venv venv
   # No Windows:
   venv\Scripts\activate
   # No Linux/macOS:
   source venv/bin/activate

2. Instalação das dependências:
   pip install -r requirements.txt

3. Download e preparação dos dados (FASE 0):
   python src/download_data.py

4. Execução dos testes unitários (FASE 1):
   python -m unittest tests/test_core.py

5. Execução em lote dos notebooks Jupyter (FASE 2):
   jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
   jupyter nbconvert --to notebook --execute --inplace 02_classify.ipynb
   jupyter nbconvert --to notebook --execute --inplace 03_clusters.ipynb
   jupyter nbconvert --to notebook --execute --inplace 04_assoc_rules.ipynb

------------------------------------------------------------------------
RESUMO DOS RESULTADOS OBTIDOS:
------------------------------------------------------------------------
- 01_eda.ipynb:
  Identificação da assimetria nas variáveis preditoras, forte correlação
  entre median_income e median_house_value (r = 0.69), teto censitário
  em $500k e desbalanceamento de classes no crédito (21.8% inadimplentes).

- 02_classify.ipynb:
  Árvore de Decisão from scratch atingiu acurácia de 89.8%, F1-Score Macro
  de 84.5% e Recall de 71.1% na detecção de inadimplentes (superando o
  baseline trivial de F1=0). Variável dominante: loan_percent_income.

- 03_clusters.ipynb:
  K-Means from scratch identificou K=4 como ponto ótimo pelo Método do
  Cotovelo e Coeficiente de Silhueta, segmentando perfeitamente a costa de
  alto padrão, o interior acessível e os polos urbanos densos.

- 04_assoc_rules.ipynb:
  Mineração Apriori from scratch identificou padrões chave com Lift > 2.8,
  provando a associação entre faixas elevadas de renda e habitações de luxo,
  além de regras de alta convicção para regiões interioranas e imóveis econômicos.

------------------------------------------------------------------------
CONFORMIDADE COM AS RESTRIÇÕES ACADÊMICAS:
------------------------------------------------------------------------
Auditoria automática realizada: ZERO imports de scikit-learn (sklearn),
scipy.cluster, PyTorch, TensorFlow ou mlxtend em todo o código fonte e nos
notebooks analíticos.
========================================================================
