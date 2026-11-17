# Registro de cambios

## v0.3 — Entrega ACA 2 (objetivos específicos 2 y 3)

### Objetivo 2: modelado, validación y optimización

- Cuaderno `notebooks/02_objetivo2_modelos.ipynb` y módulo `src/modelado.py`: partición 70/30 antes de transformar, `Pipeline` con imputación, logaritmo con signo, escalado, codificación one-hot, desviación frente a pares (solo entrenamiento) y selección por información mutua (conserva el 70 %: 61 de 88 variables).
- Ingeniería de características nueva: variables de calendario, rezagos y ventanas móviles causales por funcionario.
- Batería: modelo base de valor Z, Isolation Forest, DBSCAN y autoencoder (núcleo no supervisado) más base ingenuo, regresión logística (Ridge y Lasso), KNN, SVM, árbol, bosque aleatorio, XGBoost, LightGBM, CatBoost y red neuronal (referencia supervisada).
- Optuna (20 pruebas, TPE, semilla 42, 3 pliegues internos) con el mismo diseño para todos los modelos; validación repetida 5×2; evaluación única en la prueba.
- Ablación en tres niveles (sin variables derivadas, variables del apartado 8.4.3 y conjunto completo), SMOTE frente a pesos de clase, sensibilidad a la contaminación y análisis a nivel de funcionario.
- Respuesta a los comentarios del profesor al ACA 1: desempeño por clase y por tipo de anomalía, por tipo de producto financiero y región, y estabilidad temporal 2020-2026, con predicciones fuera de muestra; F1 de la clase anómala como métrica principal.
- 30 réplicas del generador (semillas 101 a 130) con prueba de Wilcoxon de una cola y corrección de Bonferroni (0,05/3) para H1 y H2; `generar_replica` reproduce el dataset original con la semilla 42.
- Trazabilidad: `results/experiments.csv` (una fila por corrida), `results/tables/metricas_por_particion.csv` y bitácora del objetivo 2 en `docs/tablas/tabla_bitacora_decisiones_obj2.csv`.

### Respuesta del profesor a las dudas del objetivo 2

- Sección 5.1 del cuaderno 02: contraste del `eps` de DBSCAN hallado con Optuna frente al codo del gráfico de k-distancias (sin etiquetas; k = `min_samples` de Optuna y k = 2·d, valor propuesto por Sander et al., 1998, p. 182), figura `docs/figuras/fig_kdistancia_dbscan.png` y tabla `results/tables/dbscan_eps_kdistancia.csv`.
- Sección 10.1: nivel de funcionario con predicciones fuera de muestra (3 pliegues estratificados × 20 repeticiones, referencia de pares y detector sin el funcionario evaluado), tabla `results/tables/resultados_funcionario_fuera_de_muestra.csv` y función `evaluar_funcionario_oof` en `src/modelado.py`.
- Salvedad declarada en la introducción del cuaderno: la búsqueda de hiperparámetros con la etiqueta convierte el flujo en selección de modelos supervisada (débilmente supervisado), con la prueba del 30 % intacta.
- Dos decisiones nuevas en la bitácora.

### Objetivo 3: tablero

- Cuaderno `notebooks/03_objetivo3_tablero.ipynb` y módulo `src/tablero.py`: puntaje de riesgo por solicitud y por funcionario (Isolation Forest sin etiquetas, configuración fijada de antemano), niveles de riesgo del funcionario, explicación con valores SHAP (con comprobación de aditividad), modelo dimensional (5 tablas de hechos, 8 dimensiones y 16 relaciones), 18 indicadores con fórmula y medida DAX, y validación de totales entre la fuente y el tablero (51 controles, uno por cada indicador K01 a K18 además de registros y desgloses).
- Salidas: tablas para Power BI en `data/dashboard/`, guía `dashboard/GUIA_POWER_BI.md`, hoja `dashboard/Validacion_totales_tablero.xlsx` (con columna para los valores de Power BI), tablas en `results/tables_obj3/` y figuras `docs/figuras/fig_obj3_*.png`.
- Las etiquetas sintéticas solo están en `fact_validacion_sintetica`; ninguna tabla operativa las contiene.

### Ajustes de cierre

- Corregido el conteo de variables retenidas por la selección por información mutua: son 61 de 88, no 62. El cuaderno 02 lo calcula y lo exporta a `results/tables/variables_retenidas.csv`; el informe usa ese valor. Las métricas no cambian.
- Cuaderno 02: nueva tabla `results/tables/z_aprobacion_atipicos.csv` con el valor Z de aprobación de los 6 funcionarios atípicos con dos referencias de pares (antes era un cálculo hecho fuera del cuaderno).
- Cuaderno 02: la figura del mejor modelo (`fig_obj2_mejor_modelo.png`) incluye la curva ROC con su AUC-ROC además de la matriz de confusión y la curva de precisión y exhaustividad.
- `src/figura_fases.py`: el diagrama de fases marca las fases de modelado y evaluación como completadas y el despliegue como parcial; figura regenerada en `docs/figuras/fig_8_0_fases_proyecto.png`.
- Los tiempos de ejecución documentados son: cuaderno 01 menos de 1 minuto, cuaderno 02 unos 33 minutos y cuaderno 03 unos 2 minutos (2 núcleos).

### Pendientes de esta versión

- El `.pbix` y las capturas del tablero las elabora el equipo; con ellos se completa la columna de Power BI de la validación de totales.
- Ejecución de los cuadernos 02 y 03 en Colab por el equipo para confirmar la reproducción en ese entorno.

### Publicación

- La v0.3 se publicó en GitHub y se fusionó en `main`; las entregas quedaron marcadas con las etiquetas `v0.2` y `v0.3`. Licencias confirmadas: MIT para el código y CC BY 4.0 para los datos.
- Supuestos por confirmar: modelo que alimenta el tablero (Isolation Forest sin etiquetas), variables del apartado 8.4.3 para el puntaje y cortes de los niveles de riesgo. Las medidas DAX no se pudieron ejecutar en Power BI desde el entorno de análisis.

## v0.2 — Primera entrega formal (objetivo específico 1)

- Generador reproducible del dataset sintético (`src/generar_dataset.py`, semilla 42) con salida idéntica al archivo original (SHA-256 en `data/raw/SHA256SUMS.txt`).
- Cuaderno `notebooks/01_objetivo1_dataset_ETL_EDA_FE.ipynb`: adquisición, ficha, diccionario, ETL con conteos antes y después, EDA univariado y multivariado, ingeniería de características, protocolo de evaluación y bitácora de decisiones.
- Módulo `src/features.py` con las variables derivadas por solicitud y por funcionario.
- Figuras en `docs/figuras/` (incluye el diagrama de fases del proyecto, generado con `src/figura_fases.py`) y tablas en `docs/tablas/`; salidas de datos en `data/processed/`.
