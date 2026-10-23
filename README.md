# Detección de comportamientos atípicos de funcionarios en la aprobación de créditos

Código y datos del proyecto de grado **«Modelo analítico para la detección de comportamientos atípicos de funcionarios en la aprobación de créditos del sector financiero privado colombiano»** (Especialización en Analítica de Datos, CUN).

**Autores:** Ángel Antonio Rodríguez Goyo, Julio Cesar Sánchez Jiménez y Gladys Dalexa Picón Solano. **Tutor:** Andrés Felipe Solís Pino.  
**Versión:** v0.2 (objetivo específico 1: dataset, ETL, análisis exploratorio e ingeniería de características).

## Contenido

```
├── notebooks/01_objetivo1_dataset_ETL_EDA_FE.ipynb   # cuaderno principal (se ejecuta de principio a fin)
├── src/
│   ├── generar_dataset.py       # generador del dataset sintético (semilla 42)
│   ├── features.py              # variables derivadas por solicitud y por funcionario
│   └── figura_fases.py          # diagrama de las seis fases del proyecto
├── data/
│   ├── raw/                     # dataset original + SHA256SUMS.txt + adquisicion.json
│   └── processed/               # datos limpios, matrices de características y etiquetas separadas
├── docs/figuras/                # figuras generadas por el cuaderno (PNG)
├── docs/tablas/                 # tablas generadas por el cuaderno (CSV)
├── results/resultados_obj1.json # cifras e interpretaciones generadas por el cuaderno
├── requirements.txt · LICENSE · DATA_LICENSE · CHANGELOG.md
```

## Cómo reproducirlo en Google Colab

1. Publique la carpeta como repositorio de GitHub (público) y reemplace `<USUARIO>/<REPOSITORIO>` en esta línea y en la celda 0 del cuaderno.
2. Abra el cuaderno en Colab con: `https://colab.research.google.com/github/<USUARIO>/<REPOSITORIO>/blob/main/notebooks/01_objetivo1_dataset_ETL_EDA_FE.ipynb`
3. En la primera celda de código, escriba la dirección del repositorio en `REPO_URL` (`https://github.com/<USUARIO>/<REPOSITORIO>.git`).
4. Menú **Entorno de ejecución → Ejecutar todo**. Tarda menos de un minuto. Colab ya incluye las bibliotecas necesarias.
5. Al terminar, el cuaderno habrá regenerado `data/processed/`, `docs/figuras/`, `docs/tablas/` y `results/resultados_obj1.json`. Para descargarlos: panel de archivos de Colab → carpeta `proyecto` → clic derecho → Descargar.

## Cómo reproducirlo en un equipo local

```bash
git clone https://github.com/<USUARIO>/<REPOSITORIO>.git && cd <REPOSITORIO>
python -m venv .venv && source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/01_objetivo1_dataset_ETL_EDA_FE.ipynb
```

## Verificación de reproducibilidad

- El dataset se regenera con `python src/generar_dataset.py --salida data/raw` y debe producir un archivo con la misma huella SHA-256 que figura en `data/raw/SHA256SUMS.txt`.
- El cuaderno comprueba esa huella al comenzar y se detiene si no coincide.
- Al final, `results/resultados_obj1.json` guarda la huella SHA-256 de cada archivo de `data/processed/`; dos ejecuciones deben dar las mismas huellas.

## Datos

El dataset es **sintético** (9.500 solicitudes, 220 funcionarios simulados, 2020-01-01 a 2026-06-29, 7,97 % de registros anómalos). No contiene datos personales. Las etiquetas `es_anomalia` y `tipo_anomalia` se separan de los datos de entrada y se reservan para evaluar los modelos. El diccionario de datos y la ficha del dataset están en `docs/tablas/`.

## Licencias

Código: MIT (`LICENSE`). Datos: CC BY 4.0 (`DATA_LICENSE`). Ambas son propuestas pendientes de confirmación por el equipo.
