# Registro de cambios

## v0.2 — Primera entrega formal (objetivo específico 1)

- Generador reproducible del dataset sintético (`src/generar_dataset.py`, semilla 42) con salida idéntica al archivo original (SHA-256 en `data/raw/SHA256SUMS.txt`).
- Cuaderno `notebooks/01_objetivo1_dataset_ETL_EDA_FE.ipynb`: adquisición, ficha, diccionario, ETL con conteos antes y después, EDA univariado y multivariado, ingeniería de características, protocolo de evaluación y bitácora de decisiones.
- Módulo `src/features.py` con las variables derivadas por solicitud y por funcionario.
- Figuras en `docs/figuras/` (incluye el diagrama de fases del proyecto, generado con `src/figura_fases.py`) y tablas en `docs/tablas/`; salidas de datos en `data/processed/`.
- Pendiente para v0.3 (objetivo 2): Pipeline, modelo base, Isolation Forest, DBSCAN, autoencoder y evaluación.
