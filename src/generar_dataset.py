"""
Generador del dataset sintético de solicitudes de crédito.

Proyecto: "Modelo analítico para la detección de comportamientos atípicos de
funcionarios en la aprobación de créditos del sector financiero privado
colombiano" (Especialización en Analítica de Datos, CUN).

Sigue el prompt del Anexo A del informe: 30 variables socioeconómicas,
crediticias y organizacionales más dos columnas de verdad-base
(es_anomalia, tipo_anomalia), entre 9.000 y 10.000 registros y 8 % de anomalías
controladas, con coherencia interna entre variables.

Uso:
    python src/generar_dataset.py --salida data/raw
    python src/generar_dataset.py --salida data/raw --con-xlsx

La semilla es fija (42): con las mismas versiones de numpy y pandas el CSV es
idéntico byte a byte (se verifica con data/raw/SHA256SUMS.txt).

Apoyo metodológico:
  - Aggarwal, C. C. (2017). Outlier analysis (2.ª ed.). Springer.
  - Patki, N., Wedge, R., & Veeramachaneni, K. (2016). The synthetic data vault.
  - Ley Estatutaria 1581 de 2012 (protección de datos personales, Colombia),
    razón por la que se usan datos sintéticos y no datos reales.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import date, timedelta

# ------------------------------------------------------------------
# Configuración general
# ------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

N = 9500                 # dentro del rango 9.000-10.000 registros
N_FUNCIONARIOS = 220      # número de funcionarios evaluadores
TASA_ANOMALIAS = 0.08     # 8%, dentro del rango 5-10% pedido

# ------------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------------
def clip(x, lo, hi):
    return np.clip(x, lo, hi)

def weighted_choice(categories, weights, size):
    return rng.choice(categories, size=size, p=np.array(weights) / np.sum(weights))

# ------------------------------------------------------------------
# 1. Variables demográficas y socioeconómicas
# ------------------------------------------------------------------
id_solicitante = np.arange(1, N + 1)

edad = clip(rng.normal(37, 10, N), 18, 70).round().astype(int)

genero = weighted_choice(["Masculino", "Femenino"], [0.49, 0.51], N)

nivel_educativo = weighted_choice(
    ["Primaria", "Secundaria", "Técnico", "Universitario", "Posgrado"],
    [0.05, 0.32, 0.28, 0.28, 0.07], N)

estado_civil = weighted_choice(
    ["Soltero", "Casado", "Unión libre", "Divorciado", "Viudo"],
    [0.34, 0.30, 0.24, 0.09, 0.03], N)

# Ingreso mensual: distribución sesgada hacia valores bajos (lognormal)
ingreso_mensual = clip(rng.lognormal(mean=14.9, sigma=0.55, size=N), 1_000_000, 20_000_000)

# Gastos mensuales correlacionados con el ingreso
gastos_mensuales = clip(
    ingreso_mensual * rng.uniform(0.45, 0.9, N) + rng.normal(0, 150_000, N),
    500_000, 15_000_000)

# Estrato de vivienda concentrado en 2-4
estrato_vivienda = weighted_choice([1, 2, 3, 4, 5, 6], [0.06, 0.27, 0.31, 0.22, 0.09, 0.05], N)

# Propiedad de vivienda: más probable con mayor edad/ingreso
prob_propiedad = clip(0.15 + 0.01 * (edad - 18) + (ingreso_mensual / 20_000_000) * 0.3, 0.05, 0.85)
propiedad_vivienda = np.where(rng.random(N) < prob_propiedad, "Sí", "No")

tipo_empleo = weighted_choice(
    ["Formal", "Informal", "Independiente", "Público", "Privado"],
    [0.30, 0.18, 0.22, 0.12, 0.18], N)

antiguedad_laboral = clip(
    rng.uniform(0, 1, N) * np.minimum(30, np.maximum(edad - 18, 1)), 0, 30
).round().astype(int)

# ------------------------------------------------------------------
# 2. Variables crediticias
# ------------------------------------------------------------------
score_crediticio = clip(rng.normal(650, 120, N), 0, 1000).round().astype(int)

historial_crediticio = np.select(
    [score_crediticio >= 700, score_crediticio >= 500],
    ["Bueno", "Regular"],
    default="Malo")

verificacion_centrales = np.empty(N, dtype=object)
for cat, mask, probs in [
    ("Bueno", historial_crediticio == "Bueno", [0.85, 0.05, 0.10]),
    ("Regular", historial_crediticio == "Regular", [0.45, 0.25, 0.30]),
    ("Malo", historial_crediticio == "Malo", [0.10, 0.55, 0.35]),
]:
    idx = np.where(mask)[0]
    verificacion_centrales[idx] = weighted_choice(
        ["Aprobado", "Reportado", "En seguimiento"], probs, len(idx))

numero_creditos_vigentes = clip(rng.poisson(2.2, N), 0, 10)

nivel_endeudamiento = clip(
    rng.uniform(0.1, 1.5, N) + (700 - score_crediticio) / 1000, 0.1, 1.5)

historial_mora = np.where(
    historial_crediticio == "Bueno",
    clip(rng.exponential(3, N), 0, 30),
    clip(rng.exponential(45, N), 0, 180)
).round().astype(int)

tipo_producto_financiero = weighted_choice(
    ["Consumo", "Hipotecario", "Vehículo", "Libre inversión"],
    [0.45, 0.15, 0.15, 0.25], N)

garantias_ofrecidas = np.empty(N, dtype=object)
for prod, probs in [
    ("Hipotecario", [0.05, 0.10, 0.80, 0.05]),
    ("Vehículo", [0.10, 0.75, 0.05, 0.10]),
    ("Consumo", [0.70, 0.05, 0.05, 0.20]),
    ("Libre inversión", [0.60, 0.05, 0.05, 0.30]),
]:
    idx = np.where(tipo_producto_financiero == prod)[0]
    garantias_ofrecidas[idx] = weighted_choice(
        ["Ninguna", "Vehículo", "Inmueble", "Codeudor"], probs, len(idx))

# Monto solicitado correlacionado con ingreso y tipo de producto
multiplicador_producto = np.select(
    [tipo_producto_financiero == "Hipotecario",
     tipo_producto_financiero == "Vehículo",
     tipo_producto_financiero == "Libre inversión"],
    [25, 10, 6],
    default=3)
monto_solicitado = clip(
    ingreso_mensual * multiplicador_producto * rng.lognormal(0, 0.35, N),
    500_000, 100_000_000)

plazo_solicitado = np.select(
    [tipo_producto_financiero == "Hipotecario",
     tipo_producto_financiero == "Vehículo",
     tipo_producto_financiero == "Libre inversión"],
    [clip(rng.normal(48, 8, N), 24, 60),
     clip(rng.normal(36, 8, N), 12, 60),
     clip(rng.normal(24, 8, N), 6, 48)],
    default=clip(rng.normal(18, 6, N), 6, 36)).round().astype(int)

# ------------------------------------------------------------------
# 3. Variables organizacionales
# ------------------------------------------------------------------
id_funcionario = rng.integers(1, N_FUNCIONARIOS + 1, N)

canal_solicitud = weighted_choice(
    ["Sucursal física", "Digital", "Asesor comercial"], [0.35, 0.40, 0.25], N)

segmento_cliente = weighted_choice(
    ["Nuevo", "Recurrente", "Preferencial"], [0.40, 0.45, 0.15], N)

region = weighted_choice(
    ["Bogotá", "Antioquia", "Valle", "Costa Caribe", "Santander", "Eje Cafetero", "Otras"],
    [0.28, 0.20, 0.14, 0.13, 0.09, 0.08, 0.08], N)

fecha_inicio = date(2020, 1, 1)
fecha_fin = date(2026, 6, 30)
dias_rango = (fecha_fin - fecha_inicio).days
fecha_solicitud = np.array([fecha_inicio + timedelta(days=int(d))
                             for d in rng.integers(0, dias_rango, N)])

# Ratio ingreso/monto para calcular probabilidad base de aprobación
ratio_carga = (monto_solicitado / plazo_solicitado) / ingreso_mensual
score_aprobacion = (
    (score_crediticio / 1000) * 0.5
    + (historial_crediticio == "Bueno").astype(float) * 0.2
    - (nivel_endeudamiento / 1.5) * 0.2
    - clip(ratio_carga, 0, 2) * 0.15
    - (historial_mora / 180) * 0.15
)
prob_aprobado = clip(0.5 + score_aprobacion, 0.03, 0.97)

excepciones_politicas = np.where(rng.random(N) < 0.08, "Sí", "No")

estado_aprobacion = np.where(
    rng.random(N) < prob_aprobado, "Aprobado",
    np.where(rng.random(N) < 0.4, "En estudio", "Rechazado"))

tiempo_respuesta = clip(rng.normal(8, 5, N), 1, 30).round().astype(int)

# ------------------------------------------------------------------
# 4. Inyección de anomalías controladas (5-10% => 8%)
# ------------------------------------------------------------------
n_anom = int(round(N * TASA_ANOMALIAS))
anomaly_idx = rng.choice(N, size=n_anom, replace=False)
# Dividir las anomalías en 3 arquetipos, según el procedimiento de generación
tercio = n_anom // 3
idx_monto = anomaly_idx[:tercio]
# Nota: esta muestra no se usa; el tipo "funcionario con aprobación atípica" se define por funcionario, en el bloque (b).
idx_funcionario = anomaly_idx[tercio:2 * tercio]
idx_tiempo = anomaly_idx[2 * tercio:]

es_anomalia = np.zeros(N, dtype=int)
tipo_anomalia = np.full(N, "Ninguna", dtype=object)

# (a) Montos desproporcionados respecto al ingreso
monto_solicitado[idx_monto] = clip(
    ingreso_mensual[idx_monto] * rng.uniform(25, 60, len(idx_monto)), 500_000, 100_000_000)
es_anomalia[idx_monto] = 1
tipo_anomalia[idx_monto] = "Monto desproporcionado vs. ingreso"

# (b) Funcionarios con tasas de aprobación inusualmente altas
funcionarios_atipicos = rng.choice(np.arange(1, N_FUNCIONARIOS + 1),
                                    size=max(3, int(N_FUNCIONARIOS * 0.03)), replace=False)
mask_func_atipico = np.isin(id_funcionario, funcionarios_atipicos)
# Reforzar la selección de anomalías de este tipo sobre registros de esos funcionarios
idx_funcionario_final = np.where(mask_func_atipico)[0]
if len(idx_funcionario_final) > 0:
    estado_aprobacion[idx_funcionario_final] = np.where(
        rng.random(len(idx_funcionario_final)) < 0.95, "Aprobado", estado_aprobacion[idx_funcionario_final])
    excepciones_politicas[idx_funcionario_final] = np.where(
        rng.random(len(idx_funcionario_final)) < 0.5, "Sí", excepciones_politicas[idx_funcionario_final])
    es_anomalia[idx_funcionario_final] = 1
    tipo_anomalia[idx_funcionario_final] = "Funcionario con aprobación atípica"

# (c) Tiempos de respuesta excesivos
tiempo_respuesta[idx_tiempo] = rng.integers(35, 70, len(idx_tiempo))
es_anomalia[idx_tiempo] = 1
tipo_anomalia[idx_tiempo] = "Tiempo de respuesta excesivo"

# ------------------------------------------------------------------
# 5. Número de créditos aprobados por funcionario (variable acumulada)
# ------------------------------------------------------------------
df_tmp = pd.DataFrame({"id_funcionario_evaluador": id_funcionario,
                        "estado_aprobacion": estado_aprobacion})
aprobados_por_funcionario = (
    df_tmp[df_tmp["estado_aprobacion"] == "Aprobado"]
    .groupby("id_funcionario_evaluador").size()
)
numero_creditos_aprobados_funcionario = (
    pd.Series(id_funcionario).map(aprobados_por_funcionario).fillna(0).astype(int).values
)

# ------------------------------------------------------------------
# 6. Ensamblar el DataFrame final
# ------------------------------------------------------------------
df = pd.DataFrame({
    "id_solicitante": id_solicitante,
    "edad": edad,
    "genero": genero,
    "nivel_educativo": nivel_educativo,
    "estado_civil": estado_civil,
    "ingreso_mensual": ingreso_mensual.round(0).astype(int),
    "gastos_mensuales": gastos_mensuales.round(0).astype(int),
    "estrato_vivienda": estrato_vivienda,
    "propiedad_vivienda": propiedad_vivienda,
    "tipo_empleo": tipo_empleo,
    "antiguedad_laboral": antiguedad_laboral,
    "historial_crediticio": historial_crediticio,
    "verificacion_centrales_riesgo": verificacion_centrales,
    "score_crediticio": score_crediticio,
    "numero_creditos_vigentes": numero_creditos_vigentes,
    "nivel_endeudamiento": nivel_endeudamiento.round(2),
    "historial_mora_dias": historial_mora,
    "garantias_ofrecidas": garantias_ofrecidas,
    "tipo_producto_financiero": tipo_producto_financiero,
    "monto_solicitado": monto_solicitado.round(0).astype(int),
    "plazo_solicitado_meses": plazo_solicitado,
    "estado_aprobacion": estado_aprobacion,
    "tiempo_respuesta_dias": tiempo_respuesta,
    "id_funcionario_evaluador": id_funcionario,
    "creditos_aprobados_funcionario": numero_creditos_aprobados_funcionario,
    "excepciones_politicas_internas": excepciones_politicas,
    "canal_solicitud": canal_solicitud,
    "segmento_cliente": segmento_cliente,
    "region": region,
    "fecha_solicitud": fecha_solicitud,
    # Columnas de verdad-base (SOLO para validar modelos, no usar como feature de entrada)
    "es_anomalia": es_anomalia,
    "tipo_anomalia": tipo_anomalia,
})

df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # mezclar filas
df["id_solicitante"] = np.arange(1, N + 1)  # re-numerar tras mezclar

# ------------------------------------------------------------------
# 7. Exportar
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
parser.add_argument("--salida", default=".", help="carpeta de destino")
parser.add_argument("--con-xlsx", action="store_true", help="exportar también .xlsx")
args = parser.parse_args()
salida = Path(args.salida)
salida.mkdir(parents=True, exist_ok=True)

df.to_csv(salida / "dataset_creditos_sinteticos.csv", index=False, encoding="utf-8-sig")
if args.con_xlsx:
    df.to_excel(salida / "dataset_creditos_sinteticos.xlsx", index=False, sheet_name="dataset")

print("Registros generados:", len(df))
print("Anomalías inyectadas:", df["es_anomalia"].sum(), f"({df['es_anomalia'].mean()*100:.1f}%)")
print(df["tipo_anomalia"].value_counts())
print("Archivo:", salida / "dataset_creditos_sinteticos.csv")
