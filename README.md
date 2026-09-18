# diabetes — Prevendo a incidência de Diabetes com Kedro

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

Este projeto transforma o notebook `diabetes-prediction.ipynb` em pipelines Kedro reprodutíveis. Ele segue a mesma arquitetura do projeto de churn da aula: 4 pipelines, uma API FastAPI e um container Docker.

A base é a *Pima Indians Diabetes* do Kaggle. Cada linha é uma paciente com 21 anos ou mais, e o alvo é `Outcome` (1 = diabetes).

| Arquivo em `data/01_raw/` | Linhas | Uso |
|---|---|---|
| `diabetes-dataset-modelling.csv` | 652 | Treino e avaliação |
| `diabetes-dataset-inference.csv` | 116 | Inferência (traz `Outcome`, ignorado na predição) |

## Como rodar

```bash
uv sync                                   # instala as dependências a partir do uv.lock
uv run kedro run                          # roda os 4 pipelines (33 nós)
uv run kedro run --pipeline=inference     # roda só um pipeline
uv run kedro viz                          # visualiza o DAG em http://127.0.0.1:4141
```

## Pipelines

| Pipeline | O que faz | Principais saídas |
|---|---|---|
| `data_engineering` | Limpeza → split → imputação → outliers → features → encoding → scaling | `master_table`, `modelling_*` |
| `modelling` | Baseline (LogisticRegression) e grid search (RandomForest), avaliados pela mesma função | `baseline_metrics`, `optimized_metrics` |
| `refit` | Refaz toda a cadeia em todos os splits para produção | `production_*`, `production_master_table` |
| `inference` | Só transforma e prediz, com os artefatos de produção | `inference_predictions` |

### Do notebook ao Kedro

Cada etapa com estado do notebook virou um par `fit_*` / `transform_*`:
- O `fit_*` aprende apenas com os splits listados em `split_to_fit`, que no modelling é só `train`.
- O `transform_*` aplica o que foi aprendido em todas as linhas.

O notebook ajustava tudo na base inteira **antes** do split, o que causava vazamento de dados. O `refit` e o `inference` reutilizam as mesmas funções, mudando apenas os parâmetros.

| Etapa do notebook | Nó(s) Kedro | O que mudou |
|---|---|---|
| Zeros viram NaN | `clean_data` | As colunas vêm de `columns.zero_as_missing`. `Pregnancies = 0` continua sendo um valor válido. |
| KNNImputer na base toda | `fit_imputers` / `transform_imputers` | Usa a mediana calculada **só no train**. |
| `replace_with_thresholds` | `fit_outlier_caps` / `transform_outlier_caps` | Mesma regra (q05/q95 e 1.5×IQR), mas com os limites calculados no train. |
| Features `NEW_*` | `create_features` | Sem estado, linha a linha. Dois bugs do notebook foram corrigidos (abaixo). |
| LabelEncoder + get_dummies | `fit_encoders` / `transform_encoders` | Usa OrdinalEncoder. Uma categoria nunca vista vira `-1` em vez de derrubar a API. |
| RobustScaler | `fit_scalers` / `transform_scalers` | Ajustado no train. |
| 9 modelos + GridSearch | `train_model` / `optimize_hyperparameters` | O modelo e o grid ficam no YAML (`class_path`, `param_grid`). |

**Bugs do notebook corrigidos:**
- `NEW_AGE_BMI_NOM`: a regra de obesidade usava `BMI > 18.5` e sobrescrevia as outras categorias. Agora usa as mesmas faixas exclusivas de `NEW_BMI`.
- `NEW_INSULIN_SCORE`: `set_insulin` retornava `None` para valores normais. Agora retorna `Normal` entre 16 e 166 e `Abnormal` fora dessa faixa.
- Métricas: o notebook chamava `metric(y_pred, y_test)` com os argumentos invertidos, o que trocava precision e recall. Também calculava o AUC a partir de rótulos. Agora é `metric(y_true, y_pred)`, e o AUC é calculado com `predict_proba`.

### Resultados

Split aleatório de 70/15/15 com semente 42. O grid search roda em train + test, e o `validate` fica totalmente de fora.

| Modelo | Split | Acurácia | ROC AUC | F1 macro |
|---|---|---|---|---|
| LogisticRegression (baseline) | validate | 0.824 | 0.888 | 0.801 |
| RandomForest (otimizado) | validate | 0.835 | 0.878 | 0.818 |
| RandomForest de produção (refit) | arquivo de inferência (116 linhas nunca vistas) | 0.759 | 0.826 | 0.744 |

Melhores hiperparâmetros (CV com 5 folds, `roc_auc` = 0.834): `n_estimators=200`, `max_depth=10`, `min_samples_split=10`.

Todos os valores estão em `data/08_reporting/*.json`. A última linha foi obtida comparando `inference_predictions.json` com o `Outcome` real do arquivo de inferência.

## API

```bash
uv run uvicorn diabetes.api:app --app-dir src --host 0.0.0.0 --port 8000
```

A documentação interativa fica em `/docs`.

| Método | Rota | Comportamento |
|---|---|---|
| `GET` | `/health` | Status e se os artefatos de produção existem |
| `POST` | `/train` | Roda data_engineering → modelling → refit em background e devolve um `run_id` |
| `GET` | `/train/{run_id}` | `running` / `completed` / `failed` |
| `POST` | `/inference` | Scoring síncrono a partir de JSON, sem escrever nada em disco |
| `POST` | `/batch-inference` | Roda o pipeline de inferência sobre o arquivo do catálogo, em background |

```bash
curl -s localhost:8000/health

curl -s -X POST localhost:8000/inference \
  -H 'Content-Type: application/json' \
  -d '{"instances":[{"Pregnancies":6,"Glucose":148,"BloodPressure":72,
       "SkinThickness":35,"Insulin":0,"BMI":33.6,
       "DiabetesPedigreeFunction":0.627,"Age":50}]}'
```

Um campo ausente, ou um zero em uma medida clínica, é imputado com a mediana de produção.

Dois pontos de design vêm do projeto de churn:
- Todos os handlers usam `def`, nunca `async def`, para que o trabalho pesado de CPU do Kedro rode no thread pool.
- O `/inference` injeta o payload no catálogo como `MemoryDataset`. Assim, o **mesmo** pipeline `inference` serve tanto o batch quanto o online.

## Docker

```bash
uv run kedro run                  # gera os artefatos de produção em data/06_models
docker compose up --build -d
curl -s localhost:8000/health
docker compose down
```

- `conf/` é montado como somente leitura e `data/` como leitura e escrita, por isso nada disso é copiado para a imagem.
- Se a porta 8000 estiver ocupada, use `API_PORT=8001 docker compose up -d`.
- Sem artefatos, o `/inference` responde 409. Nesse caso, chame `POST /train` pelo container.

## Testes

```bash
uv run pytest               # 40 testes: nós, DAG e API (~91% de cobertura)
uv run ruff check src tests
```

Os testes de nós verificam:
- que o imputer, os limites de outlier, os encoders e os scalers usam só o split de treino (testes de vazamento);
- os dois bugs corrigidos do notebook (testes de regressão).

### Problema no macOS

O `uv` grava os arquivos `.pth` da instalação editável com a flag `UF_HIDDEN` do macOS, e o Python 3.14 ignora `.pth` ocultos. Por isso, `import diabetes` pode falhar mesmo com o pacote instalado.

O `kedro run` e o `pytest` não são afetados: o Kedro e o `pythonpath = ["src"]` já colocam `src/` no path. No uvicorn, o `--app-dir src` faz o mesmo. Para corrigir a flag diretamente:

```bash
chflags nohidden .venv/lib/python*/site-packages/*.pth
```

## Estrutura

```
conf/base/catalog.yml        # todo o I/O (camadas 01_raw … 08_reporting)
conf/base/parameters.yml     # colunas, split, cortes clínicos, modelos e grids
src/diabetes/pipelines/      # data_engineering, modelling, refit, inference
src/diabetes/api/            # FastAPI: main (rotas), schemas (Pydantic), service (Kedro)
tests/                       # unitários (nós), integração (DAG) e e2e (API)
```
