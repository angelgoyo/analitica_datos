"""Ingeniería de características del proyecto (objetivo específico 1).

Dos niveles de análisis, según el apartado 7.2 del informe:

* Nivel de solicitud: razones financieras calculadas fila a fila.
  No aprenden nada de los datos, por lo que se pueden aplicar antes de
  separar entrenamiento y prueba sin riesgo de fuga de información.
* Nivel de funcionario: indicadores agregados por funcionario evaluador y su
  desviación frente al grupo de pares. La agregación no usa la etiqueta de
  anomalía. La desviación frente a pares SÍ aprende estadísticos (media y
  desviación estándar del grupo); por eso `desviacion_pares` recibe una
  referencia explícita. En el modelado (objetivo 2) esa referencia se calculará solo con el
  conjunto de entrenamiento, dentro de un Pipeline.

Ninguna función de este módulo lee las columnas `es_anomalia` ni
`tipo_anomalia`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COL_FUNC = "id_funcionario_evaluador"

# Columnas de etiqueta: reservadas para evaluar, nunca como entrada.
COLUMNAS_ETIQUETA = ["es_anomalia", "tipo_anomalia"]


def razones_solicitud(df: pd.DataFrame) -> pd.DataFrame:
    """Variables derivadas por solicitud (sin parámetros aprendidos).

    Devuelve un DataFrame con el índice de `df` y las columnas:

    - razon_monto_ingreso: monto solicitado / ingreso mensual.
    - razon_gasto_ingreso: gastos mensuales / ingreso mensual.
    - capacidad_residual: ingreso mensual - gastos mensuales (COP).
    - cuota_estimada: monto solicitado / plazo en meses (sin intereses, COP).
    - razon_cuota_ingreso: cuota estimada / ingreso mensual.
    - razon_cuota_capacidad: cuota estimada / capacidad residual positiva
      (NaN cuando la capacidad residual no es positiva).
    - excepcion_bin: 1 si hubo excepción a políticas internas, 0 si no.
    - aprobado_bin: 1 si el estado es "Aprobado", 0 en otro caso.
    - decidido_bin: 1 si el estado es "Aprobado" o "Rechazado" (descarta "En estudio").
    - mes, anio: componentes de la fecha de solicitud.
    """
    ingreso = df["ingreso_mensual"].astype(float)
    capacidad = ingreso - df["gastos_mensuales"].astype(float)
    cuota = df["monto_solicitado"].astype(float) / df["plazo_solicitado_meses"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["razon_monto_ingreso"] = df["monto_solicitado"] / ingreso
    out["razon_gasto_ingreso"] = df["gastos_mensuales"] / ingreso
    out["capacidad_residual"] = capacidad
    out["cuota_estimada"] = cuota
    out["razon_cuota_ingreso"] = cuota / ingreso
    out["razon_cuota_capacidad"] = np.where(capacidad > 0, cuota / capacidad.where(capacidad > 0), np.nan)
    out["excepcion_bin"] = (df["excepciones_politicas_internas"] == "Sí").astype(int)
    out["aprobado_bin"] = (df["estado_aprobacion"] == "Aprobado").astype(int)
    out["decidido_bin"] = df["estado_aprobacion"].isin(["Aprobado", "Rechazado"]).astype(int)
    fecha = pd.to_datetime(df["fecha_solicitud"])
    out["mes"] = fecha.dt.month
    out["anio"] = fecha.dt.year
    return out


def perfil_funcionario(df: pd.DataFrame, razones: pd.DataFrame | None = None) -> pd.DataFrame:
    """Indicadores agregados por funcionario evaluador (sin etiqueta).

    Una fila por funcionario. Indicadores:

    - n_solicitudes: solicitudes atendidas.
    - tasa_aprobacion: aprobadas / atendidas.
    - tasa_aprobacion_decididas: aprobadas / (aprobadas + rechazadas).
    - tasa_excepciones: proporción de solicitudes con excepción a políticas.
    - monto_promedio, monto_mediano: monto solicitado de sus solicitudes (COP).
    - razon_monto_ingreso_prom: promedio de monto / ingreso.
    - tiempo_resp_prom, tiempo_resp_p90: días de respuesta (media y percentil 90).
    - score_prom: score crediticio promedio de los solicitantes que evalúa.
    - mora_prom: mora promedio (días) de los solicitantes que evalúa.
    """
    r = razones if razones is not None else razones_solicitud(df)
    t = pd.DataFrame(
        {
            COL_FUNC: df[COL_FUNC].values,
            "aprobado": r["aprobado_bin"].values,
            "decidido": r["decidido_bin"].values,
            "excepcion": r["excepcion_bin"].values,
            "monto": df["monto_solicitado"].values,
            "razon_mi": r["razon_monto_ingreso"].values,
            "tiempo": df["tiempo_respuesta_dias"].values,
            "score": df["score_crediticio"].values,
            "mora": df["historial_mora_dias"].values,
        }
    )
    g = t.groupby(COL_FUNC)
    p = pd.DataFrame(
        {
            "n_solicitudes": g.size(),
            "tasa_aprobacion": g["aprobado"].mean(),
            "tasa_aprobacion_decididas": g["aprobado"].sum() / g["decidido"].sum(),
            "tasa_excepciones": g["excepcion"].mean(),
            "monto_promedio": g["monto"].mean(),
            "monto_mediano": g["monto"].median(),
            "razon_monto_ingreso_prom": g["razon_mi"].mean(),
            "tiempo_resp_prom": g["tiempo"].mean(),
            "tiempo_resp_p90": g["tiempo"].quantile(0.9),
            "score_prom": g["score"].mean(),
            "mora_prom": g["mora"].mean(),
        }
    )
    return p


INDICADORES_PARES = [
    "tasa_aprobacion",
    "tasa_excepciones",
    "monto_promedio",
    "razon_monto_ingreso_prom",
    "tiempo_resp_prom",
]


def desviacion_pares(perfil: pd.DataFrame, referencia: pd.DataFrame | None = None,
                     columnas: list[str] | None = None) -> pd.DataFrame:
    """Puntaje Z de cada funcionario frente al grupo de pares.

    z = (indicador - media_pares) / desviación_pares. Los estadísticos del
    grupo salen de `referencia` (por defecto, el propio `perfil`). Para evitar
    fuga de información al modelar, `referencia` debe ser el perfil calculado
    solo con datos de entrenamiento.
    """
    cols = columnas or INDICADORES_PARES
    ref = referencia if referencia is not None else perfil
    mu = ref[cols].mean()
    sd = ref[cols].std(ddof=0).replace(0, np.nan)
    z = (perfil[cols] - mu) / sd
    z.columns = [f"z_{c}" for c in cols]
    return z
