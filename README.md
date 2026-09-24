# diabetes — Prevendo a incidência de Diabetes com Kedro

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

**Autor:** Luiz Felipe Possebon, exercício de avaliação da disciplina de Deploy (Insper PADS, 3º trimestre).

Este projeto transforma o notebook `diabetes-prediction.ipynb` em pipelines Kedro reprodutíveis. Ele segue a mesma arquitetura do projeto de churn da aula: 4 pipelines, uma API FastAPI e um container Docker.

A base é a *Pima Indians Diabetes* do Kaggle. Cada linha é uma paciente com 21 anos ou mais, e o alvo é `Outcome` (1 = diabetes).

| Arquivo em `data/01_raw/` | Linhas | Uso |
|---|---|---|
| `diabetes-dataset-modelling.csv` | 652 | Treino e avaliação |
| `diabetes-dataset-inference.csv` | 116 | Inferência (traz `Outcome`, ignorado na predição) |

## Como rodar

```bash
uv sync                                   # instala as dependências a partir do uv.lock
uv run kedro run                          # roda os 4 pipelines (35 nós)
uv run kedro run --pipelines=inference    # roda só um pipeline
uv run kedro viz                          # visualiza o DAG em http://127.0.0.1:4141
```

## Pipelines

| Pipeline | O que faz | Principais saídas |
|---|---|---|
| `data_engineering` | Limpeza → split → imputação → outliers → features → encoding → scaling | `master_table`, `modelling_*` |
| `modelling` | Baseline (LogisticRegression) e grid search (RandomForest), avaliados pela mesma função. Escolhe o campeão e registra a curva de corte | `baseline_metrics`, `optimized_metrics`, `champion_report`, `threshold_curve` |
| `refit` | Refaz toda a cadeia em todos os splits e retreina o **campeão** para produção | `production_*`, `production_master_table` |
| `inference` | Só transforma e prediz, com os artefatos de produção | `inference_predictions` |

### Do notebook ao Kedro

Cada etapa com estado do notebook virou um par `fit_*` / `transform_*`:
- O `fit_*` aprende apenas com os splits listados em `split_to_fit`, que no modelling é só `train`.
- O `transform_*` aplica o que foi aprendido em todas as linhas.

O notebook ajustava tudo na base inteira **antes** do split, o que causava vazamento de dados. O `refit` e o `inference` reutilizam as mesmas funções, mudando apenas os parâmetros.

| Etapa do notebook | Nó(s) Kedro | O que mudou |
|---|---|---|
| Zeros viram NaN | `clean_data` | As colunas vêm de `columns.zero_as_missing`. `Pregnancies = 0` continua sendo um valor válido. Insulin (48% ausente) e SkinThickness (29%) ganham uma flag `<COL>_MISSING` antes da imputação. |
| KNNImputer na base toda | `fit_imputers` / `transform_imputers` | Usa a mediana calculada **só no train**. |
| `replace_with_thresholds` | `fit_outlier_caps` / `transform_outlier_caps` | Mesma regra (q05/q95 e 1.5×IQR), mas com os limites calculados no train. |
| Features `NEW_*` | `create_features` | Sem estado, linha a linha. Dois bugs do notebook foram corrigidos (abaixo). |
| LabelEncoder + get_dummies | `fit_encoders` / `transform_encoders` | Usa OneHotEncoder, como o `get_dummies`: as categorias são nominais, e um código inteiro daria à regressão logística uma ordem que não existe. Uma categoria nunca vista vira uma linha de zeros em vez de derrubar a API. |
| RobustScaler | `fit_scalers` / `transform_scalers` | Ajustado no train. |
| 9 modelos + GridSearch | `train_model` / `optimize_hyperparameters` | O modelo e o grid ficam no YAML (`class_path`, `param_grid`). |
| Escolha do melhor modelo | `select_champion` | Regra explícita no YAML, em vez de promover sempre o otimizado (abaixo). |

**Bugs do notebook corrigidos:**
- `NEW_AGE_BMI_NOM`: a regra de obesidade usava `BMI > 18.5` e sobrescrevia as outras categorias. Agora usa as mesmas faixas exclusivas de `NEW_BMI`.
- `NEW_INSULIN_SCORE`: `set_insulin` retornava `None` para valores normais. Agora retorna `Normal` entre 16 e 166 e `Abnormal` fora dessa faixa.
- Métricas: o notebook chamava `metric(y_pred, y_test)` com os argumentos invertidos, o que trocava precision e recall. Também calculava o AUC a partir de rótulos. Agora é `metric(y_true, y_pred)`, e o AUC é calculado com `predict_proba`.

### Parâmetros

As decisões ficam em [`conf/base/parameters.yml`](conf/base/parameters.yml) e a mecânica fica no código, seguindo o padrão do projeto de churn. Cada nó recebe os parâmetros como entradas explícitas (`params:<bloco>`) e nunca lê a configuração por conta própria.

| Bloco | O que decide |
|---|---|
| `columns` | Alvo, medidas brutas, colunas onde 0 significa ausente, flags de ausência, features numéricas e categóricas |
| `split` | Proporções de train/test/validate, semente e estratificação pelo alvo |
| `outliers`, `feature_engineering` | Quantis e fator do IQR, faixas clínicas de IMC, glicose, idade e insulina |
| `modelling_*` / `refit_*` | Colunas de cada artefato e os splits usados no ajuste (`split_to_fit`) |
| `modelling_baseline`, `modelling_optimization` | Modelo (`class_path`), `init_args`, grid, `cv`, `scoring`, `n_jobs` |
| `evaluation` | Bootstrap do intervalo de confiança do AUC |
| `champion_selection` | Split, métrica e margem mínima para o otimizado substituir o baseline |
| `threshold_analysis` | Folds e cortes avaliados na curva out-of-fold |
| `decision` | **Corte de negócio**: uma paciente é sinalizada quando P(diabetes) ≥ `threshold` |

### Resultados

Split estratificado de 70/15/15 com semente 42: os três splits mantêm a prevalência de ~35% de diabéticas. O grid search roda em train + test, e o `validate` fica totalmente de fora. O AUC não depende do corte e vem com IC de 95% por bootstrap; as demais métricas usam `decision.threshold = 0.35`.

| Modelo | Dados | ROC AUC [IC 95%] | Recall | Precisão | Não detectadas |
|---|---|---|---|---|---|
| LogisticRegression (baseline) | validate (97) | 0.811 [0.714–0.894] | 0.76 | 0.55 | 8 de 34 |
| RandomForest (otimizado, **campeão**) | validate (97) | 0.838 [0.757–0.909] | 0.85 | 0.59 | 5 de 34 |
| RandomForest de produção (refit) | arquivo de inferência (116 nunca vistas) | 0.825 [0.738–0.899] | 0.83 | 0.59 | 7 de 40 |

Melhores hiperparâmetros (CV com 5 folds, `roc_auc` = 0.844): `n_estimators=200`, `max_depth=5`, `min_samples_split=5`.

As duas primeiras linhas, incluindo a matriz de confusão, estão em `data/08_reporting/*_metrics.json`. Cada split traz `in_sample`, que é `true` quando o modelo foi ajustado nele. A última linha foi obtida comparando `inference_predictions.json` com o `Outcome` real do arquivo de inferência.

**Como ler esses números.** Os intervalos têm cerca de 0.18 de largura: com ~100 pacientes por holdout, um único split é ruidoso. Antes da estratificação, o validate sorteava outras pacientes e dava AUC entre 0.88 e 0.89 para os mesmos tipos de modelo. No arquivo de inferência, que nenhuma dessas mudanças toca, o resultado ficou igual ao anterior (0.825 contra 0.826). O valor mais estável é o da validação cruzada, em torno de 0.84.

**Escolha do campeão.** O `select_champion` compara os dois modelos no `validate` e só promove o otimizado se ele superar o baseline por mais de `champion_selection.min_improvement` (hoje 0.0). Em caso de empate, fica o baseline, que é menor e interpretável. O resultado vai para `champion_report.json`. O nó recusa comparar em um split usado no ajuste de qualquer um dos modelos.

**One-hot e flags de ausência.** Uma ablação com CV repetida 5×5 nas 652 linhas mostrou efeito neutro no AUC (−0.002, dentro do ruído). As duas mudanças ficaram por correção metodológica: a regressão logística não recebe mais uma ordem falsa entre categorias, e uma insulina não medida deixa de ser indistinguível de uma medida na mediana. Para desligar as flags, basta esvaziar `columns.missing_indicator`.

**Por que 0.35 e não 0.5.** Deixar passar uma diabética custa muito mais do que um exame confirmatório a mais. O corte foi escolhido com predições out-of-fold só nos dados de modelagem, sem olhar o arquivo de inferência. A curva completa fica em `threshold_curve.json`, gerada pelo nó `analyse_thresholds`. Nas 116 pacientes nunca vistas, sair de 0.5 para 0.35 muda o seguinte:
- Recall: 0.68 → 0.83.
- Diabéticas não detectadas: 13 → 7.
- Pacientes enviadas ao exame: 35% → 48%.

Para mudar o equilíbrio entre casos perdidos e alarmes falsos, basta editar `decision.threshold`. O batch e a API passam a usar o novo corte sem mudança de código.

### Versionamento

Todos os modelos, predições e relatórios (`data/06_models`, `07_model_output` e `08_reporting`) são `versioned: true` no catálogo. Cada execução grava em `<arquivo>/<timestamp>/`, e a leitura pega sempre a versão mais recente. Um retreino nunca sobrescreve o modelo em serviço, e qualquer versão anterior pode ser recarregada:

```bash
uv run kedro run --pipelines=inference --load-versions=production_model:2026-09-24T22.36.12.050Z
```

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
| `GET` | `/predictions` | Devolve o dataset `inference_predictions` do último batch, lido pelo catálogo |
| `GET` | `/metrics/{baseline\|optimized}` | Devolve o dataset de métricas do modelo, por split, com `in_sample` e IC do AUC |
| `GET` | `/reports/{champion\|threshold_curve}` | Devolve a escolha do campeão ou a curva de corte out-of-fold |

```bash
curl -s localhost:8000/health

curl -s -X POST localhost:8000/inference \
  -H 'Content-Type: application/json' \
  -d '{"instances":[{"Pregnancies":6,"Glucose":148,"BloodPressure":72,
       "SkinThickness":35,"Insulin":0,"BMI":33.6,
       "DiabetesPedigreeFunction":0.627,"Age":50}]}'

curl -s localhost:8000/predictions            # dataset gerado pelo pipeline de inferência
curl -s localhost:8000/metrics/optimized      # métricas por split, no corte de negócio
```

Cada paciente é validada pelo schema `Patient` antes de chegar ao Kedro:
- `Glucose`, `BMI` e `Age` são obrigatórios, porque são os sinais mais fortes do modelo. Sem eles, a paciente seria a "mediana" imputada, que fica acima do corte e seria sinalizada.
- As demais medidas podem faltar, vir como `null` ou como `0` ("não medido", como no CSV). Nesses casos são imputadas com a mediana de produção.
- Valores fora de faixas plausíveis, `Age` abaixo de 21 (fora da população de treino) e campos desconhecidos, como `glucose` minúsculo, são rejeitados com 422.

As rotas `/predictions` e `/metrics` só leem, e passam pelo catálogo Kedro, nunca pelo caminho do arquivo. O formato e a localização continuam definidos no `catalog.yml`. Elas atendem apenas a uma lista fechada de datasets (`inference_predictions`, `baseline_metrics` e `optimized_metrics`). Modelos em pickle e tabelas intermediárias com dados de pacientes nunca saem pela API.

Concorrência:
- Um `POST /train` com outro treino em andamento responde 409. O mesmo vale para `/batch-inference`.
- Um lock protege os artefatos de produção. O `refit` e o batch o seguram enquanto gravam ou leem, e o `/inference` só enquanto carrega os cinco artefatos em memória. Uma predição nunca mistura imputers novos com um modelo antigo, nem lê um pickle pela metade.
- O lock vale dentro do processo da API. Ele não cobre um `kedro run` disparado pelo terminal enquanto a API está servindo.

Um erro interno responde 500 com uma mensagem genérica. O traceback vai só para o log do servidor.

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
- O `CMD` chama o `uvicorn` direto do venv da imagem. Com `uv run`, o container reinstalava as dependências de dev a cada subida e falhava sem internet. A imagem foi testada com `--network none`.
- `pandas` e `numpy` são dependências declaradas de produção. Antes, só chegavam pelo grupo de dev.

## Testes

```bash
uv run pytest               # 79 testes: nós, DAG, ponta a ponta e API (~96% de cobertura)
uv run ruff check src tests
```

Os testes verificam:
- que o imputer, os limites de outlier, os encoders e os scalers usam só o split de treino (testes de vazamento);
- os dois bugs corrigidos do notebook (testes de regressão);
- a regra do campeão, a flag `in_sample` e a curva de corte;
- os 4 pipelines rodando de ponta a ponta sobre os CSVs reais, em memória (`tests/test_end_to_end.py`);
- a validação da API, o 409 para treino concorrente e o lock dos artefatos.

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
