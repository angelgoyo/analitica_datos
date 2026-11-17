# Detección de comportamientos atípicos de funcionarios en la aprobación de créditos

Código y datos del proyecto de grado **«Modelo analítico para la detección de comportamientos atípicos de funcionarios en la aprobación de créditos del sector financiero privado colombiano»** (Especialización en Analítica de Datos, CUN).

**Autores:** Ángel Antonio Rodríguez Goyo, Julio Cesar Sánchez Jiménez y Gladys Dalexa Picón Solano. **Tutor:** Andrés Felipe Solís Pino.  
**Versión:** v0.3, correspondiente a la entrega ACA 2 (objetivos específicos 2 y 3). El objetivo específico 1 quedó en v0.2 (ACA 1). El objetivo específico 4 (recomendaciones) se entregará en una versión posterior.

## Estado por objetivo

| Objetivo específico | Cuaderno | Apartado del informe | Estado |
|---|---|---|---|
| 1. Dataset sintético, ETL, EDA e ingeniería de características | `01_objetivo1_dataset_ETL_EDA_FE.ipynb` | 8.1 a 8.4 | Entregado en v0.2 |
| 2. Modelado, validación y optimización | `02_objetivo2_modelos.ipynb` | 8.5 y 8.6 | Completo en v0.3 |
| 3. Tablero: puntaje, SHAP, modelo dimensional, indicadores y validación de totales | `03_objetivo3_tablero.ipynb` | 8.7 | Completo en el código; el `.pbix` y la columna de Power BI de la validación de totales los completa el equipo |
| 4. Recomendaciones | — | 8.8 | Pendiente (siguiente entrega) |

## Contenido

```
├── notebooks/01_objetivo1_dataset_ETL_EDA_FE.ipynb   # objetivo 1: dataset, ETL, EDA e ingeniería de características
├── notebooks/02_objetivo2_modelos.ipynb              # objetivo 2: modelos, validación 5×2, Optuna, ablación, desglose por tipo/segmento/año y 30 réplicas (se ejecuta de principio a fin)
├── notebooks/03_objetivo3_tablero.ipynb              # objetivo 3: puntaje, explicación SHAP, modelo dimensional, indicadores y validación de totales
├── src/
│   ├── generar_dataset.py       # generador del dataset sintético (semilla 42)
│   ├── features.py              # variables derivadas por solicitud y por funcionario
│   ├── modelado.py              # Pipeline, detectores, batería de modelos, Optuna y validación repetida
│   ├── tablero.py               # puntaje de riesgo, SHAP, modelo dimensional, KPIs y validación de totales (objetivo 3)
│   └── figura_fases.py          # diagrama de las seis fases del proyecto
├── data/
│   ├── raw/                     # dataset original + SHA256SUMS.txt + adquisicion.json
│   ├── processed/               # datos limpios, matrices de características y etiquetas separadas
│   └── dashboard/               # tablas de hechos y dimensiones que se cargan en Power BI
├── docs/figuras/                # figuras generadas por los cuadernos (PNG)
├── docs/tablas/                 # tablas generadas por los cuadernos (CSV), incluidas las bitácoras de decisiones
├── results/resultados_obj1.json # cifras e interpretaciones del cuaderno 01
├── results/resultados_obj2.json # cifras e interpretaciones del cuaderno 02
├── results/resultados_obj3.json # cifras e interpretaciones del cuaderno 03
├── results/tables/              # tablas de resultados del objetivo 2 (métricas por partición, Optuna, ablación, réplicas, Wilcoxon, variables retenidas, etc.)
├── results/tables_obj3/         # tablas de resultados del objetivo 3 (indicadores, relaciones, SHAP, validación)
├── results/experiments.csv      # una fila por corrida (modelo, hiperparámetros, métricas, semilla, versión)
├── dashboard/                   # guía de Power BI, maquetas y hoja de validación de totales
├── requirements.txt · LICENSE · DATA_LICENSE · CHANGELOG.md
```

Las cifras que aparecen en el informe salen de los archivos de `results/` y de `docs/tablas/`; el informe no contiene valores escritos a mano. Cada tabla o figura del informe cita en su nota el cuaderno y la sección que la genera.

## Cómo reproducirlo en Google Colab

Ejecute los cuadernos en este orden. Cada uno clona este repositorio por sí mismo, instala lo que le falta y termina con el mensaje «Todas las comprobaciones pasaron.».

| Cuaderno | Colab | Tiempo aproximado |
|---|---|---|
| 01 | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/angelgoyo/analitica_datos/blob/main/notebooks/01_objetivo1_dataset_ETL_EDA_FE.ipynb) | menos de 1 minuto |
| 02 | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/angelgoyo/analitica_datos/blob/main/notebooks/02_objetivo2_modelos.ipynb) | unos 33 minutos con 2 núcleos |
| 03 | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/angelgoyo/analitica_datos/blob/main/notebooks/03_objetivo3_tablero.ipynb) | unos 2 minutos |

1. Abra el cuaderno con el botón de la tabla y use el menú **Entorno de ejecución → Ejecutar todo**.
2. **Cuaderno 01.** Al final debe aparecer «Todas las comprobaciones pasaron.» y, en la celda que regenera el dataset, «Coincide con el original: True». Regenera `data/processed/`, `docs/figuras/`, `docs/tablas/` y `results/resultados_obj1.json`.
3. **Cuaderno 02.** Instala Optuna, XGBoost, LightGBM, CatBoost e imbalanced-learn. La mayor parte del tiempo corresponde a la optimización con Optuna; las 30 réplicas del generador agregan unos 2 a 3 minutos. Deja las tablas en `results/tables/`, las figuras en `docs/figuras/` y el registro de corridas en `results/experiments.csv`.
4. **Cuaderno 03.** Instala `shap` y `openpyxl`. Deja las tablas para Power BI en `data/dashboard/`, los resultados en `results/tables_obj3/`, la hoja `dashboard/Validacion_totales_tablero.xlsx` y la guía `dashboard/GUIA_POWER_BI.md`. El archivo `.pbix` lo arma el equipo.
5. Para descargar los resultados: panel de archivos de Colab → carpeta `proyecto` → clic derecho → Descargar.

## Cómo reproducirlo en un equipo local

```bash
git clone https://github.com/angelgoyo/analitica_datos.git && cd analitica_datos
python -m venv .venv && source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/01_objetivo1_dataset_ETL_EDA_FE.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks/02_objetivo2_modelos.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/03_objetivo3_tablero.ipynb
```

## Verificación de reproducibilidad

- El dataset se regenera con `python src/generar_dataset.py --salida data/raw` y debe producir un archivo con la misma huella SHA-256 que figura en `data/raw/SHA256SUMS.txt`.
- El cuaderno 01 comprueba esa huella al comenzar y se detiene si no coincide.
- `results/resultados_obj1.json` guarda la huella SHA-256 de cada archivo de `data/processed/`; dos ejecuciones deben dar las mismas huellas.
- Todas las semillas están fijadas (42 para el dataset y los modelos; 101 a 130 para las 30 réplicas del generador). Con las mismas versiones de las bibliotecas (`requirements.txt`), las métricas del cuaderno 02 se repiten; los tiempos de ejecución sí varían entre equipos.
- Los cuadernos 02 y 03 contienen comprobaciones propias (por ejemplo, que la fila de Optuna del contraste de k-distancias coincida con la evaluación final, la aditividad de los valores SHAP y las 27 pruebas de integridad del modelo dimensional) y se detienen si alguna falla.

## Datos

El dataset es **sintético** (9.500 solicitudes, 220 funcionarios simulados, 2020-01-01 a 2026-06-29, 7,97 % de registros anómalos). No contiene datos personales. Las etiquetas `es_anomalia` y `tipo_anomalia` se separan de los datos de entrada y se reservan para evaluar los modelos; en el modelo dimensional solo están en `fact_validacion_sintetica`. El diccionario de datos y la ficha del dataset están en `docs/tablas/`.

## Alcance y salvedades

- Los resultados son evidencia dentro del escenario sintético y no una prueba de desempeño sobre datos reales.
- La búsqueda de hiperparámetros con Optuna usa la etiqueta en validación, de modo que el flujo es una selección de modelos supervisada; el conjunto de prueba (30 %) permanece intacto. El informe lo declara.
- Los tipos de anomalía inyectada se interpretan como aproximaciones a conductas de elusión de controles y colusión, no como réplicas de casos documentados.

## Licencias

Código: licencia MIT (`LICENSE`). Datos sintéticos: Creative Commons Atribución 4.0 Internacional, CC BY 4.0 (`DATA_LICENSE`).

## Versiones

Las entregas están marcadas con etiquetas de Git: `v0.2` (ACA 1, objetivo 1) y `v0.3` (ACA 2, objetivos 2 y 3). La rama `main` contiene la última versión.
