# diabetes — Risco de diabetes em 5 anos com Kedro

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

**Autor:** Luiz Felipe Possebon, exercício de avaliação da disciplina de Deploy (Insper PADS, 3º trimestre).

Pipelines Kedro reprodutíveis para estimar o risco de diabetes a partir da análise do notebook `diabetes-prediction.ipynb`. A arquitetura segue a do projeto de churn da aula: 4 pipelines, uma API FastAPI e um container Docker.

A base é a *Pima Indians Diabetes* do Kaggle (NIDDK; Smith et al., 1988). Cada linha é uma mulher indígena Pima do Arizona com 21 anos ou mais, e o alvo é `Outcome` (1 = diabetes).

| Arquivo em`data/01_raw/`         | Linhas | Uso                                                   |
| ---------------------------------- | ------ | ----------------------------------------------------- |
| `diabetes-dataset-modelling.csv` | 652    | Treino e avaliação                                  |
| `diabetes-dataset-inference.csv` | 116    | Inferência (traz`Outcome`, ignorado na predição) |

## Pergunta clínica

> Qual é o risco de uma mulher **não diabética e não gestante**, de 21 a 81 anos, que acabou de fazer um teste oral de tolerância à glicose (TOTG, 75 g), **desenvolver diabetes em até 5 anos**?

O modelo é **prognóstico**, não diagnóstico. Os motivos:

- `Glucose` e `Insulin` são valores **2 horas após a sobrecarga** do TOTG, e o desfecho é o critério da OMS para diabetes (glicose 2h ≥ 200 mg/dL) em um exame posterior.
- Nenhuma paciente da base tem glicose ≥ 200 (o máximo é 199). Quem já está nesse valor tem o desfecho, e não um risco dele.
- Um modelo diagnóstico seria circular: para usá-lo, seria preciso ter feito o TOTG, e o TOTG já é o exame diagnóstico.

Cada paciente recebe uma de quatro respostas:

| Situação | Resposta |
| --- | --- |
| Glicose 2h ≥ 200, glicemia de jejum ≥ 126 mg/dL ou HbA1c ≥ 6,5% | `decision_basis = "diagnostic_criterion"`: já preenche um critério de diabetes da ADA, listado em `criteria_met`. O modelo não é usado. A conduta é confirmar (a ADA pede um segundo exame alterado na paciente assintomática) e tratar. |
| Risco ≥ `decision.threshold` | `prediction = 1`, `decision_basis = "model"`: alto risco em 5 anos. Encaminhar para prevenção (mudança de estilo de vida) e glicemia anual. |
| Glicose 2h de 140 a 199 mg/dL e risco abaixo do corte | `prediction = 1`, `decision_basis = "impaired_glucose_tolerance"`: tolerância diminuída à glicose (pré-diabetes). A diretriz já encaminha para prevenção, e o modelo não a desautoriza. |
| Risco abaixo do corte e glicose 2h < 140 | `prediction = 0`: seguimento habitual. |

**Escopo.** O modelo vale só para mulheres de 21 a 81 anos, a faixa da base de treino, que não estejam grávidas, não tenham diabetes conhecido e não usem medicamento que altere a glicose. A API recusa pedidos fora disso. Ele **não foi validado externamente** e não é um dispositivo diagnóstico (ver [Limitações](#limitações-e-validação-externa)).

## Como rodar

```bash
uv sync                                   # instala as dependências a partir do uv.lock
uv run kedro run                          # roda os 4 pipelines (38 nós)
uv run kedro run --pipelines=inference    # roda só um pipeline
uv run kedro viz                          # visualiza o DAG em http://127.0.0.1:4141
```

## Pipelines

| Pipeline             | O que faz                                                                                                                                                                                    | Principais saídas                                                                    |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `data_engineering` | Limpeza → split → imputação → outliers → features → encoding → scaling                                                                                                               | `master_table`, `modelling_*`                                                     |
| `modelling`        | Baseline (regressão logística com 4 variáveis clínicas) e desafiante (RandomForest com grid search), avaliados pela mesma função. Escolhe o campeão e registra a análise de decisão, comparando o modelo com a regra da diretriz | `baseline_metrics`, `optimized_metrics`, `champion_report`, `threshold_curve` |
| `refit`            | Refaz toda a cadeia em todos os splits, retreina o**campeão** para produção e publica os odds ratios                                                                                | `production_*`, `production_master_table`, `production_odds_ratios`             |
| `inference`        | Só transforma e prediz, com os artefatos de produção. Depois aplica a regra da diretriz (tolerância diminuída) e os critérios diagnósticos                                               | `inference_predictions`                                                             |

## Decisões de desenho

### Sem vazamento de dados

Toda etapa que aprende algo com os dados é um par `fit_*` / `transform_*`:

- O `fit_*` aprende apenas com os splits listados em `split_to_fit`. No modelling, isso é só o `train`.
- O `transform_*` aplica o que foi aprendido em todas as linhas.

Assim, nenhuma estatística do test ou do validate chega ao modelo durante a avaliação. O `refit` e o `inference` reutilizam as mesmas funções, mudando apenas os parâmetros.

| Etapa              | Nó(s) Kedro                                      | Critério                                                                                                                                                                                                                                     |
| ------------------ | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Valores ausentes   | `clean_data`                                    | Um zero em`columns.zero_as_missing` é fisiologicamente impossível e significa "não medido". `Pregnancies = 0` é um valor válido. Insulin (48% ausente) e SkinThickness (29%) ganham uma flag `<COL>_MISSING` antes da imputação. |
| Imputação        | `fit_imputers` / `transform_imputers`         | Mediana calculada **só no train**.                                                                                                                                                                                                    |
| Outliers           | `fit_outlier_caps` / `transform_outlier_caps` | Limites q05/q95 ± 1.5×IQR, calculados no train.                                                                                                                                                                                             |
| Features clínicas | `create_features`                               | Sem estado, linha a linha, com os cortes das diretrizes (abaixo).                                                                                                                                                                             |
| Encoding           | `fit_encoders` / `transform_encoders`         | OneHotEncoder: as categorias são nominais, e um código inteiro daria a um modelo linear uma ordem que não existe. Uma categoria nunca vista vira uma linha de zeros em vez de derrubar a API.                                              |
| Escala             | `fit_scalers` / `transform_scalers`           | RobustScaler (mediana e IQR), ajustado no train. Serve bem a medidas clínicas assimétricas.                                                                                                                                                 |
| Modelos            | `train_model` / `optimize_hyperparameters`    | O modelo, as variáveis (`features`) e o grid ficam no YAML (`class_path`, `param_grid`).                                                                                                                                               |
| Campeão           | `select_champion`                               | Regra estatística explícita (abaixo).                                                                                                                                                                                                       |

### Critérios clínicos das features

- **Glicose.** `NEW_GLUCOSE` e `NEW_AGE_GLUCOSE_NOM` usam os cortes do TOTG para a glicose 2h: < 140 normal, 140–199 tolerância diminuída (pré-diabetes), ≥ 200 diabetes. Os cortes de glicemia de jejum (100/126) não se aplicam a essa medida: às 2 horas, 126 mg/dL é um valor normal.
- **IMC.** `NEW_BMI` segue as classes da OMS: < 18.5 baixo peso, 18.5–24.9 normal, 25–29.9 sobrepeso, ≥ 30 obesidade.
- **Faixas fechadas à esquerda.** As diretrizes definem os cortes como limites inferiores ("IMC ≥ 30", "glicose ≥ 140"). Por isso, um IMC de 25 já é sobrepeso e uma glicose de 140 já é tolerância diminuída.
- **Idade.** `NEW_AGE_CAT` separa as pacientes com 50 anos ou mais. `NEW_AGE_BMI_NOM` e `NEW_AGE_GLUCOSE_NOM` cruzam a idade com as mesmas faixas exclusivas de IMC e de glicose.

Três candidatas não são usadas, por não terem base fisiológica para valores de 2h:

- **Escore de insulina "normal/anormal".** A classe "anormal" juntaria insulina baixa (falência da célula β) e alta (resistência à insulina), dois estados opostos. Além disso, as ~48% de pacientes sem insulina medida receberiam a mediana e cairiam todas em "normal".
- **Glicose × insulina.** Lembra o HOMA-IR, mas o HOMA-IR é definido com valores de jejum. Para metade das pacientes, seria só a glicose vezes a mediana imputada.
- **Glicose × gestações.** Não há mecanismo que justifique essa interação.

A `DiabetesPedigreeFunction` do CSV também fica de fora, por outro motivo: ela não existe na prática clínica. É uma fórmula própria de Smith et al. que exige o heredograma com as idades de cada parente, e nenhum consultório a calcula. Com ela no modelo, cada paciente real receberia a mediana imputada. A perda é pequena e fica dentro do ruído: AUC de 0.838 com ela e 0.834 sem ela em CV repetida no train + test, e 0.824 contra 0.806 nas 116 pacientes nunca vistas. O histórico familiar é um fator de risco real, porém. Quando houver uma coorte que registre "parente de 1º grau com diabetes (sim/não)", como no FINDRISC, essa é a variável a acrescentar.

As métricas seguem a convenção `metric(y_true, y_pred)`, e o AUC é sempre calculado a partir das probabilidades (`predict_proba`), nunca dos rótulos.

### Parâmetros

As decisões ficam em [`conf/base/parameters.yml`](conf/base/parameters.yml) e a mecânica fica no código, seguindo o padrão do projeto de churn. Cada nó recebe os parâmetros como entradas explícitas (`params:<bloco>`) e nunca lê a configuração por conta própria.

| Bloco                                              | O que decide                                                                                                   |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `columns`                                        | Alvo, medidas brutas, colunas onde 0 significa ausente, flags de ausência, features numéricas e categóricas |
| `split`                                          | Proporções de train/test/validate, semente e estratificação pelo alvo                                      |
| `outliers`, `feature_engineering`              | Quantis e fator do IQR; classes de IMC da OMS e cortes da glicose 2h do TOTG                                   |
| `modelling_*` / `refit_*`                      | Colunas de cada artefato e os splits usados no ajuste (`split_to_fit`)                                       |
| `modelling_baseline`, `modelling_optimization` | Modelo (`class_path`), variáveis (`features`), `init_args`, grid, `cv`, `scoring`, `n_jobs`       |
| `evaluation`                                     | Bootstrap do intervalo de confiança do AUC                                                                    |
| `champion_selection`                             | Split e bootstrap pareado que decidem se o desafiante substitui o baseline                                     |
| `threshold_analysis`                             | Folds, cortes avaliados e prevalências de referência da análise de decisão                                 |
| `guideline_referral`                             | Glicose 2h a partir da qual a diretriz encaminha para prevenção (140 mg/dL), com ou sem o modelo           |
| `diagnostic_criteria`                            | Critérios da ADA que já definem diabetes: glicose 2h ≥ 200, jejum ≥ 126 mg/dL, HbA1c ≥ 6,5%                 |
| `decision`                                       | **Corte clínico**: uma paciente é sinalizada quando o risco em 5 anos ≥ `threshold`                 |

## Resultados

O split é estratificado em 70/15/15, com semente 42, e os três splits mantêm a prevalência de ~35% de diabéticas. O baseline treina só no train. O grid search do desafiante roda em train + test. O `validate` fica totalmente de fora nos dois casos. O AUC não depende do corte e vem com IC de 95% por bootstrap. As demais métricas usam `decision.threshold = 0.30`.

| Modelo                                                   | Dados                                     | ROC AUC [IC 95%]     | Sensib. | Especif. | VPP  | VPN  | Não detectadas |
| -------------------------------------------------------- | ----------------------------------------- | -------------------- | ------- | -------- | ---- | ---- | --------------- |
| Regressão logística, 4 variáveis (**campeão**) | validate (97)                             | 0.856 [0.778–0.925] | 0.85    | 0.67     | 0.58 | 0.89 | 5 de 34         |
| RandomForest, 28 features (desafiante)                   | validate (97)                             | 0.829 [0.744–0.903] | 0.88    | 0.57     | 0.53 | 0.90 | 4 de 34         |
| Regressão logística de produção (refit)              | arquivo de inferência (116 nunca vistas) | 0.806 [0.723–0.885] | 0.78    | 0.66     | 0.54 | 0.85 | 9 de 40         |
| Regra servida: modelo **ou** tolerância diminuída    | arquivo de inferência (116 nunca vistas) | —                    | 0.78    | 0.64     | 0.53 | 0.84 | 9 de 40         |

**Variáveis do baseline.** São quatro, todas fatores de risco estabelecidos e registrados em qualquer consulta: glicose 2h, IMC, idade e número de gestações. A escolha foi feita com CV 5-fold repetida 20 vezes em train + test, sem olhar o validate: AUC de 0.834, contra 0.831 sem gestações e 0.829 com as 7 medidas brutas. Com a `DiabetesPedigreeFunction` seria 0.838, mas ela não pode ser obtida na prática (ver [Critérios clínicos das features](#critérios-clínicos-das-features)).

**Complexidade.** O RandomForest, com 28 features e grid search, não supera a regressão logística com 4 variáveis. Ele tem AUC de 0.918 nos próprios dados de ajuste (train + test) e 0.829 no validate, um sinal de overfitting.

**Calibração.** Um escore de risco só é útil se a probabilidade puder ser lida como risco. Na base Pima, a regressão logística é bem calibrada:

- No validate: Brier 0.148, razão observado/esperado (O/E) 0.90, inclinação 1.08.
- No arquivo de inferência: Brier 0.170, O/E 0.92.

Essa calibração vale para uma população em que ~35% das mulheres desenvolvem diabetes em 5 anos. Com uma incidência menor, a ordenação das pacientes se mantém, mas a probabilidade superestima o risco absoluto:

| Incidência em 5 anos | 0.30 exibido corresponde a | 0.50 exibido corresponde a |
| -------------------- | -------------------------- | -------------------------- |
| 35% (base Pima)      | 0.30                       | 0.50                       |
| 15%                  | 0.12                       | 0.25                       |
| 10%                  | 0.08                       | 0.17                       |

Por isso a API descreve `probability` como calibrada para a coorte de desenvolvimento. Antes de ela ser lida como risco absoluto em outra população, o intercepto precisa ser recalibrado com dados locais.

Os números de validate, incluindo matriz de confusão, especificidade, VPN e calibração, estão em `data/08_reporting/*_metrics.json`. Cada split traz `in_sample`, que é `true` quando o modelo foi ajustado nele. A última linha da tabela vem da comparação de `inference_predictions.json` com o `Outcome` real do arquivo de inferência.

### Escolha do campeão

Em ~100 pacientes, comparar só os valores pontuais do AUC promoveria o desafiante por qualquer diferença, mesmo dentro do ruído. Por isso, o `select_champion` usa um critério estatístico:

1. Pontua os dois modelos nas **mesmas** pacientes do validate.
2. Reamostra a diferença de AUC (desafiante − baseline) com 2000 réplicas de bootstrap pareado. A reamostragem pareada cancela o quanto cada paciente é fácil ou difícil de classificar, algo que dois intervalos separados não fazem.
3. Promove o desafiante só se o limite inferior do IC 95% dessa diferença passar de `min_improvement`. Caso contrário, o baseline é o campeão, por ser menor e interpretável.

Nesta execução, a diferença foi de −0.028, com IC [−0.069; 0.013], e o baseline é o campeão. O relatório fica em `champion_report.json`. O nó recusa comparar em um split usado no ajuste de qualquer um dos modelos.

### Odds ratios

O `refit` publica `production_odds_ratios.json`, com os coeficientes em unidades clínicas. O modelo vê `(x − mediana) / IQR`, então cada coeficiente é um log-OR por IQR, e dividir pelo IQR dá o OR por unidade. Valores do modelo de produção desta execução:

| Variável                | OR por unidade                         | OR por IQR (IQR)     |
| ------------------------ | -------------------------------------- | -------------------- |
| Glicose 2h               | 1.036 por mg/dL (1.42 a cada 10 mg/dL) | 3.93 (39 mg/dL)      |
| IMC                      | 1.099 por kg/m²                       | 2.36 (9.1 kg/m²)    |
| Idade                    | 1.026 por ano                          | 1.50 (16 anos)       |
| Gestações              | 1.091 por gestação                   | 1.55 (5 gestações) |

As estimativas têm penalização L2 (padrão do sklearn, C = 1), portanto ficam levemente puxadas para 1. Um campeão sem coeficientes, como um RandomForest, gera o relatório com `odds_ratios = null` em vez de falhar.

### One-hot e flags de ausência

Só o desafiante usa essas colunas. Uma ablação com CV repetida 5×5 nas 652 linhas mostrou efeito neutro no AUC (−0.002, dentro do ruído). Mesmo assim, as duas são mantidas por critério metodológico: nenhuma categoria recebe uma ordem falsa, e uma insulina não medida não se confunde com uma medida na mediana. Para desligar as flags, basta esvaziar `columns.missing_indicator`.

## O corte de 0.30 e a curva de decisão

O `analyse_thresholds` refaz o campeão com CV 5-fold nas 457 pacientes em que ele foi treinado, sem tocar o validate nem o arquivo de inferência. Para cada corte, ele registra em `threshold_curve.json`:

- sensibilidade, especificidade, VPP, VPN e pacientes sinalizadas;
- o **benefício líquido** da curva de decisão (Vickers & Elkin, 2006);
- VPP e VPN recalculados em prevalências mais baixas.

| Corte          | Sensib.        | Especif.       | Sinalizadas   | Benefício líquido | Encaminhar todas | VPP / VPN a 10%       | VPP / VPN a 5%        |
| -------------- | -------------- | -------------- | ------------- | ------------------- | ---------------- | --------------------- | --------------------- |
| 0.20           | 0.90           | 0.56           | 60%           | 0.244               | 0.188            | 0.19 / 0.98           | 0.10 / 0.99           |
| 0.25           | 0.86           | 0.65           | 53%           | 0.225               | 0.133            | 0.22 / 0.98           | 0.12 / 0.99           |
| **0.30** | **0.79** | **0.71** | **47%** | **0.196**     | **0.072**  | **0.23 / 0.97** | **0.12 / 0.98** |
| 0.35           | 0.71           | 0.77           | 40%           | 0.166               | 0.000            | 0.25 / 0.96           | 0.14 / 0.98           |
| 0.50           | 0.56           | 0.89           | 27%           | 0.120               | −0.300          | 0.35 / 0.95           | 0.20 / 0.97           |

Os critérios por trás do corte:

- **O corte é uma escolha clínica.** Um corte *p* diz que encontrar uma futura diabética vale (1 − *p*) / *p* encaminhamentos desnecessários. Em 0.30, isso dá cerca de 2.3.
- **O encaminhamento é barato e seguro:** prevenção por estilo de vida e glicemia anual. Por isso, perder um caso pesa mais do que alguns encaminhamentos extras.
- **O modelo vale a pena em todos os cortes de 0.10 a 0.60.** Nessa faixa, o benefício líquido dele supera tanto o de encaminhar todas quanto o de não encaminhar nenhuma (zero).
- **0.30 equilibra os dois lados:** detecta ~80% das futuras diabéticas e sinaliza ~47% das pacientes. Com 0.35, a sensibilidade cairia para 0.71. Com 0.25, as sinalizadas subiriam para 53%.
- **O VPP cai com a prevalência.** Os ~35% desta amostra estão muito acima de uma população geral. A 10% de prevalência, só ~1 em cada 4 sinalizadas desenvolveria diabetes. O VPN, por outro lado, fica acima de 0.96: o modelo é mais útil para **tranquilizar** quem fica abaixo do corte do que para confirmar quem fica acima.

Nas 116 pacientes nunca vistas:

| Corte          | Sensibilidade  | Não detectadas   | Sinalizadas   |
| -------------- | -------------- | ----------------- | ------------- |
| 0.50           | 0.60           | 16 de 40          | 34%           |
| 0.35           | 0.70           | 12 de 40          | 43%           |
| **0.30** | **0.78** | **9 de 40** | **49%** |

Para escolher outro equilíbrio entre casos perdidos e encaminhamentos, basta editar `decision.threshold`. O batch e a API usam o valor configurado, sem mudança de código.

### Modelo × diretriz

Superar "encaminhar todas" é uma barreira baixa. Sem o modelo, o clínico já encaminha para prevenção toda paciente com tolerância diminuída (glicose 2h de 140 a 199). Por isso o `analyse_thresholds` também pontua, nas mesmas pacientes e com o mesmo peso de um caso perdido (corte 0.30), a regra da diretriz sozinha e a regra que a inferência de fato serve: modelo **ou** diretriz.

| Estratégia                       | Dados                    | Sensib. | Especif. | Sinalizadas | Não detectadas | Benefício líquido |
| -------------------------------- | ------------------------ | ------- | -------- | ----------- | -------------- | ----------------- |
| Diretriz: glicose 2h ≥ 140       | out-of-fold (457)        | 0.46    | 0.90     | 23%         | 86 de 160      | 0.134             |
| Modelo: risco ≥ 0.30             | out-of-fold (457)        | 0.79    | 0.71     | 47%         | 33 de 160      | 0.196             |
| **Modelo ou diretriz** (servida) | out-of-fold (457)        | 0.79    | 0.70     | 47%         | 33 de 160      | 0.194             |
| Diretriz: glicose 2h ≥ 140       | inferência (116)         | 0.57    | 0.84     | 30%         | 17 de 40       | 0.154             |
| Modelo: risco ≥ 0.30             | inferência (116)         | 0.78    | 0.66     | 49%         | 9 de 40        | 0.171             |
| **Modelo ou diretriz** (servida) | inferência (116)         | 0.78    | 0.64     | 50%         | 9 de 40        | 0.167             |

O que a tabela mostra:

- **O modelo acrescenta algo à diretriz.** Cerca de metade das futuras diabéticas tinha glicose 2h normal (< 140) no exame, e a diretriz não encaminha nenhuma delas. O modelo detecta a maioria. Parte do ganho, porém, vem de encaminhar mais pacientes (47% contra 23%), e não só de ordenar melhor.
- **A regra combinada não custa nada e não acrescenta casos nesta base.** Quase toda paciente com tolerância diminuída já tem risco ≥ 0.30. A regra existe por coerência clínica: o modelo nunca deve desautorizar um encaminhamento que a diretriz já faz. No arquivo de inferência, só uma paciente foi sinalizada pela diretriz e não pelo modelo.

## Limitações e validação externa

- **População.** Mulheres indígenas Pima do Arizona, um dos grupos com maior prevalência de diabetes tipo 2 do mundo, com dados coletados há décadas. O modelo não foi testado em outras etnias, em homens, nem na população brasileira.
- **Sem validação externa.** Todo o desempenho acima é de validação interna: holdout e CV na mesma base. Antes de qualquer uso real, o modelo precisa ser validado em uma coorte brasileira com TOTG, como o ELSA-Brasil. Nessa coorte, é preciso medir discriminação, calibração e benefício líquido, e provavelmente recalibrar o intercepto para a prevalência local.
- **Amostra pequena.** 652 pacientes e 228 eventos. Cada holdout tem ~100 pacientes, e os ICs do AUC têm ~0.15 de largura. No arquivo de inferência, cada diabética não detectada move 2.5 pontos na sensibilidade.
- **Calibração absoluta.** A probabilidade só é um risco absoluto numa população com a incidência Pima (~35% em 5 anos). Em outra população, ela ordena as pacientes, mas superestima o risco até ser recalibrada (ver [Calibração](#resultados)).
- **Desfecho.** Na base, o diabetes foi definido só pela glicose 2h (critério da OMS da época). Hoje o diagnóstico também usa glicemia de jejum e HbA1c, então o modelo prevê um desfecho mais estreito do que o que a clínica diagnostica.
- **Entradas.** O modelo exige um TOTG, exame mais caro e demorado que a glicemia de jejum ou a HbA1c. Ele serve para estratificar o risco de quem já fez o TOTG, não como triagem populacional. Para esse fim existem escores sem exame de laboratório, como o FINDRISC. No Brasil, o TOTG de 75 g é feito sobretudo no pré-natal, justamente onde o modelo não se aplica: a API recusa gestantes.
- **Exclusões só na API.** O schema da API exige que gestação, diabetes conhecido e uso de medicação que altere a glicose sejam respondidos e negados. O batch confia no arquivo: o CSV da coorte não traz esses campos (as participantes não tinham diabetes no exame, mas a gestação não foi registrada).

## Relato (TRIPOD+AI, resumo)

| Item             | Neste projeto                                                                                                                          |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Fonte dos dados  | Pima Indians Diabetes (NIDDK; Smith et al., 1988)                                                                                      |
| Participantes    | 768 mulheres Pima de 21 a 81 anos, sem diabetes no exame. 652 para desenvolvimento, 116 reservadas para inferência                    |
| Desfecho         | Diabetes pelo critério da OMS (glicose 2h ≥ 200 mg/dL) em até 5 anos                                                                |
| Preditores       | Glicose 2h do TOTG, IMC, idade e número de gestações, todos medidos no mesmo exame.`DiabetesPedigreeFunction` excluída por não ser obtível na clínica |
| Tamanho amostral | Treino do baseline: 457 pacientes e 160 eventos (40 por preditor). Refit: 652 e 228 (57 por preditor)                                  |
| Dados ausentes   | Zero fisiologicamente impossível tratado como ausente. Mediana do treino, com flags de ausência para insulina e prega cutânea       |
| Modelo           | Regressão logística (L2, C = 1) com RobustScaler. Desafiante: RandomForest com grid search. Campeão por bootstrap pareado           |
| Desempenho       | Discriminação (AUC com IC 95%), calibração (Brier, O/E, inclinação), sensibilidade, especificidade, VPP e VPN, curva de decisão, comparação com a regra da diretriz |
| Validação      | Interna: split estratificado 70/15/15 e CV.**Não há validação externa**                                                      |
| Interpretação  | Odds ratios por unidade clínica em`production_odds_ratios.json`                                                                     |

## Versionamento

Todos os modelos, predições e relatórios (`data/06_models`, `07_model_output` e `08_reporting`) são `versioned: true` no catálogo. Cada execução grava em `<arquivo>/<timestamp>/`, e a leitura pega sempre a versão mais recente. Um retreino nunca sobrescreve o modelo em serviço, e qualquer versão anterior pode ser recarregada:

```bash
uv run kedro run --pipelines=inference --load-versions=production_model:2026-09-24T22.36.12.050Z
```

## API

```bash
uv run uvicorn diabetes.api:app --app-dir src --host 0.0.0.0 --port 8000
```

A documentação interativa fica em `/docs`.

| Método  | Rota                                                | Comportamento                                                                                                        |
| -------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `GET`  | `/health`                                         | Status e se os artefatos de produção existem                                                                       |
| `POST` | `/train`                                          | Roda data_engineering → modelling → refit em background e devolve um`run_id`                                     |
| `GET`  | `/train/{run_id}`                                 | `running` / `completed` / `failed`                                                                             |
| `POST` | `/inference`                                      | Scoring síncrono a partir de JSON, sem escrever nada em disco                                                       |
| `POST` | `/batch-inference`                                | Roda o pipeline de inferência sobre o arquivo do catálogo, em background                                           |
| `GET`  | `/predictions`                                    | Devolve o dataset`inference_predictions` do último batch, lido pelo catálogo                                     |
| `GET`  | `/metrics/{baseline\|optimized}`                   | Devolve o dataset de métricas do modelo, por split, com`in_sample`, IC do AUC, especificidade, VPN e calibração |
| `GET`  | `/reports/{champion\|threshold_curve\|odds_ratios}` | Devolve a escolha do campeão, a análise de decisão out-of-fold ou os odds ratios do modelo de produção          |

```bash
curl -s localhost:8000/health

curl -s -X POST localhost:8000/inference \
  -H 'Content-Type: application/json' \
  -d '{"instances":[{"Sex":"female","Pregnant":false,"KnownDiabetes":false,
       "GlucoseAffectingMedication":false,"Pregnancies":6,"Glucose":128,
       "BloodPressure":72,"SkinThickness":35,"Insulin":0,"BMI":33.6,"Age":50,
       "FastingGlucose":98}]}'

curl -s localhost:8000/predictions            # dataset gerado pelo pipeline de inferência
curl -s localhost:8000/metrics/baseline       # métricas por split, no corte clínico
curl -s localhost:8000/reports/odds_ratios    # odds ratios do modelo de produção
```

Cada predição traz:

- `probability`: o risco estimado de diabetes em 5 anos, calibrado para a coorte Pima (ver [Calibração](#resultados)).
- `prediction`: 1 quando a paciente é sinalizada.
- `decision_basis`:
  - `"model"`: a decisão vem do corte aplicado ao risco.
  - `"impaired_glucose_tolerance"`: glicose 2h de 140 a 199 mg/dL com risco abaixo do corte. A paciente é sinalizada pela diretriz, e a probabilidade é mantida.
  - `"diagnostic_criterion"`: um critério de diabetes da ADA foi atingido. Nesse caso, `prediction = 1`, `probability = null`, e o modelo não é usado.
- `criteria_met`: os critérios atingidos, por exemplo `["HbA1c >= 6.5"]`, para o clínico saber qual exame confirmar. Vazio nos demais casos.

Cada paciente é validada pelo schema `Patient` antes de chegar ao Kedro:

- `Sex` é obrigatório e só aceita `"female"`, porque o modelo foi desenvolvido só com mulheres.
- `Pregnant`, `KnownDiabetes` e `GlucoseAffectingMedication` (hipoglicemiante, metformina inclusive, ou glicocorticoide sistêmico) são obrigatórios e precisam ser `false`. Obrigatórios de propósito: um valor padrão presumiria a elegibilidade sem que ninguém tivesse perguntado. `true` é recusado com 422 e com o motivo clínico (na gestação, por exemplo, o TOTG é lido pelos critérios IADPSG).
- `Glucose`, `BMI` e `Age` são obrigatórios, porque são os sinais mais fortes do modelo. Sem eles, o escore descreveria a paciente "mediana" imputada, e não a paciente avaliada.
- `FastingGlucose` (mg/dL) e `HbA1c` (%) são opcionais e não entram no modelo. Servem só para o critério diagnóstico: uma paciente com jejum ≥ 126 e glicose 2h de 170 já é diabética, embora a glicose 2h sozinha não mostre isso.
- As demais medidas podem faltar, vir como `null` ou como `0` ("não medido", como no CSV). Nesses casos são imputadas com a mediana de produção.
- `DiabetesPedigreeFunction` não é aceita, porque o modelo não a usa.
- `Age` fora de 21–81 (a faixa de treino), valores fora de faixas plausíveis (como HbA1c em mmol/mol) e campos desconhecidos, como `glucose` minúsculo, são rejeitados com 422.

As rotas `/predictions`, `/metrics` e `/reports` só leem, e passam pelo catálogo Kedro, nunca pelo caminho do arquivo. O formato e a localização continuam definidos no `catalog.yml`. Elas atendem apenas a uma lista fechada de datasets: `inference_predictions`, as métricas, o relatório do campeão, a curva de decisão e os odds ratios. Modelos em pickle e tabelas intermediárias com dados de pacientes nunca saem pela API.

Concorrência:

- Um `POST /train` com outro treino em andamento responde 409. O mesmo vale para `/batch-inference`.
- Um lock protege os artefatos de produção. O `refit` e o batch o seguram enquanto gravam ou leem, e o `/inference` só enquanto carrega os cinco artefatos em memória. Uma predição nunca mistura imputers novos com um modelo antigo, nem lê um pickle pela metade.
- O lock vale dentro do processo da API. Ele não cobre um `kedro run` disparado pelo terminal enquanto a API está servindo.

Um erro interno responde 500 com uma mensagem genérica. O traceback vai só para o log do servidor.

Dois pontos de design vêm do projeto de churn:

- Todos os handlers usam `def`, nunca `async def`, para que o trabalho pesado de CPU do Kedro rode no thread pool.
- O `/inference` injeta o payload no catálogo como `MemoryDataset`. Assim, o **mesmo** pipeline `inference` serve tanto o batch quanto o online, incluindo a regra da diretriz e os critérios diagnósticos.

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
- O `CMD` chama o `uvicorn` direto do venv da imagem, sem `uv run`. Assim, o container não instala nada ao subir e funciona sem internet. A imagem foi testada com `--network none`.
- A imagem instala só as dependências de produção (`--no-dev`), por isso `pandas` e `numpy` estão declarados nelas.

## Testes

```bash
uv run pytest               # 139 testes: nós, DAG, ponta a ponta e API (~96% de cobertura)
uv run ruff check src tests
```

Os testes verificam:

- que o imputer, os limites de outlier, os encoders e os scalers usam só o split de treino (testes de vazamento);
- os cortes clínicos nas bordas (glicose 139.9/140/200, IMC 24.95/25/30), o 126 mg/dL como glicose 2h normal e que as features sem base fisiológica não são construídas;
- a regra do campeão por bootstrap pareado, incluindo um ganho pontual dentro do ruído que **não** promove o desafiante;
- a análise de decisão (benefício líquido, VPP e VPN por prevalência), a comparação com a regra da diretriz e os odds ratios em unidades clínicas;
- os critérios diagnósticos (glicose 2h ≥ 200, jejum ≥ 126, HbA1c ≥ 6,5, com os critérios atingidos listados) e a regra da diretriz (toda glicose 2h de 140 a 199 é sinalizada, e o modelo nunca retira um encaminhamento). Um valor ausente ou `0` não atinge nenhuma regra;
- que a `DiabetesPedigreeFunction` do CSV não chega ao modelo;
- os 4 pipelines rodando de ponta a ponta sobre os CSVs reais, em memória (`tests/test_end_to_end.py`);
- a validação da API (sexo, faixa etária, gestação, diabetes conhecido, medicação, exames diagnósticos), o 409 para treino concorrente e o lock dos artefatos.

### Problema no macOS

O `uv` grava os arquivos `.pth` da instalação editável com a flag `UF_HIDDEN` do macOS, e o Python 3.14 ignora `.pth` ocultos. Por isso, `import diabetes` pode falhar mesmo com o pacote instalado.

O `kedro run` e o `pytest` não são afetados: o Kedro e o `pythonpath = ["src"]` já colocam `src/` no path. No uvicorn, o `--app-dir src` faz o mesmo. Para corrigir a flag diretamente:

```bash
chflags nohidden .venv/lib/python*/site-packages/*.pth
```

## Estrutura

```
conf/base/catalog.yml        # todo o I/O (camadas 01_raw … 08_reporting)
conf/base/parameters.yml     # escopo clínico, colunas, split, cortes clínicos, modelos e grids
src/diabetes/pipelines/      # data_engineering, modelling, refit, inference
src/diabetes/api/            # FastAPI: main (rotas), schemas (Pydantic), service (Kedro)
tests/                       # unitários (nós), integração (DAG) e e2e (API)
```
