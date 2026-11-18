Leia integralmente o arquivo `SPEC.md` na raiz deste projeto. Como a pasta `data/` ainda está vazia, você será responsável também por buscar e descarregar os datasets públicos exigidos antes de iniciar a implementação.

Execute o projeto em 4 fases sequenciais:

### FASE 0: Download e Preparação dos Datasets (`data/`)
1. Instale `kagglehub` (e `requests` se necessário) no ambiente Python.
2. Crie e execute um script `src/download_data.py` para descarregar automaticamente para a pasta `data/` **2 datasets públicos do Kaggle** cobrindo rigorosamente os temas **"Housing"** e **"Home credit"** exigidos no `SPEC.md`:
   - **Housing:** `camnugent/california-housing-prices` (ou `yasserh/housing-prices-dataset`).
   - **Home Credit:** pesquise e baixe via `kagglehub` um dataset público de Home Credit / Credit Risk (ex.: `home-credit-default-risk` ou mirror público equivalente no Kaggle; se o download do Kaggle exigir autenticação para a competição específica, use automaticamente um mirror público do OpenML / GitHub do dataset Home Credit `application_train` ou dataset público de Home Credit no Kaggle).
3. Salve os CSVs finais em `data/housing.csv` e `data/home_credit.csv` (se o dataset de Home Credit tiver mais de 15.000 linhas, gere também uma amostra estratificada reprodutível `data/home_credit_sample.csv` com `random_state=42` para garantir execução rápida nos algoritmos em NumPy puro, documentando isso).
4. Registe as URLs exatas de origem de ambos os datasets num dicionário/constante para serem citadas obrigatoriamente no `01_eda.ipynb`.

### FASE 1: Módulos Core (`src/`)
1. Implemente:
   - `src/preprocessing.py`: limpeza, encoding, scaling e `train_test_split` em NumPy/Pandas puro.
   - `src/metrics.py`: matriz de confusão, accuracy, precision, recall, F1-score, WCSS e silhouette score em NumPy puro.
   - `src/decision_tree.py`: algoritmo Decision Tree escrito 100% from scratch (apenas Python/NumPy), vetorizado, suportando obrigatoriamente `max_depth`, `min_samples_split`, `min_samples_leaf` e `criterion` (`gini`/`entropy`).
   - `src/kmeans.py`: algoritmo K-Means escrito 100% from scratch (apenas Python/NumPy), com seed reprodutível, cálculo de WCSS e histórico de convergência.
2. Escreva e rode testes unitários rápidos provando que `DecisionTreeClassifier` e `KMeans` convergem, respeitam os hiperparâmetros e não importam nenhuma biblioteca proibida (`sklearn`, etc.).

### FASE 2: Construção e Execução dos 4 Notebooks
Crie e preencha `01_eda.ipynb`, `02_classify.ipynb`, `03_clusters.ipynb` e `04_assoc_rules.ipynb` seguindo todas as regras do `SPEC.md`:
- Célula inicial de Introdução com identificação do grupo (mantendo placeholders para o número de estudante e colegas), contexto do Projeto 02 e contribuição individual.
- Fonte exata dos datasets e metadados completos em `01_eda.ipynb`.
- Objetivos de Negócio e Preparação de Dados nos notebooks 02, 03 e 04.
- Registo explícito de **Resultados Intermédios** (baseline, grid search de hiperparâmetros da Decision Tree, curva $K=2..10$ no K-Means, variação de suporte/confiança no Apriori) e **Resultados Finais** com gráficos.
- Secção final de **Conclusão** em TODOS os notebooks analisando as métricas e destacando as **Lições Aprendidas**.
- Execute todos via CLI (`jupyter nbconvert --to notebook --execute --inplace`) até rodarem sem erros com todos os outputs salvos.

### FASE 3: Auditoria Final e Empacotamento
1. Audite via script/grep que não existe nenhum import de `sklearn` ou bibliotecas de ML em `src/`, `02_classify.ipynb` e `03_clusters.ipynb`.
2. Gere `requirements.txt` e `readme.txt` na raiz com o link do repositório GitHub (`https://github.com/LucasMagalhaes-sys/IA26_P02_G51233`).
3. Gere o arquivo compactado `IA_P02_G51233.zip` contendo os notebooks executados, `src/`, `data/`, `requirements.txt` e `readme.txt`.

Comece agora executando a FASE 0 e confirme quais arquivos CSV foram salvos em `data/` e suas respectivas URLs de origem.