"""Objetivo específico 3: insumos del tablero de monitoreo (modelo dimensional, KPIs y validación de totales).

Este módulo no entrena modelos nuevos: aplica el detector Isolation Forest (sin etiquetas)
a las solicitudes y a los perfiles de los funcionarios, explica el puntaje con valores SHAP
y arma las tablas del modelo dimensional que se cargan en Power BI.

Decisiones de diseño (supuestos hasta que el tutor y el profesor respondan):

* El puntaje usa Isolation Forest con configuración fijada de antemano (no ajustada con la
  etiqueta), porque es continuo, tiene explicación SHAP exacta para árboles y no depende de la
  decisión pendiente sobre el uso de etiquetas en la optimización. Cambiar el modelo implica
  cambiar `CONFIG_SOLICITUD` / `CONFIG_FUNCIONARIO` y volver a ejecutar el cuaderno 03.
* A nivel de solicitud el modelo se ajusta con el 70 % de entrenamiento (misma partición del
  cuaderno 02) y se puntúan las 9.500 solicitudes; la columna `particion` indica el origen.
* A nivel de funcionario el perfil se calcula con todas las solicitudes (es el perfil de
  monitoreo; el modelo no ve etiquetas) y la referencia de pares es el conjunto de funcionarios.
* Las etiquetas del dataset sintético NO entran en las tablas operativas. Viven solo en
  `fact_validacion_sintetica`, que sirve para la página de validación del tablero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import modelado as M

SEED = M.SEED
CONFIG_SOLICITUD = dict(n_estimators=200, max_samples=256, max_features=1.0)
CONFIG_FUNCIONARIO = dict(n_estimators=200, max_samples=1.0, max_features=1.0)
FE_SOLICITUD = "8.4.3"                          # variables del apartado 8.4.3 (sin calendario ni ventanas móviles); True = conjunto completo
CONTAMINACION = M.CONTAMINACION_BASE            # 7,5 %: proporción de solicitudes o funcionarios que se marca
CORTES_NIVEL = {"Medio": 0.75, "Alto": 1 - CONTAMINACION, "Crítico": 0.975}   # cuantiles del puntaje de funcionarios

# --------------------------------------------------------------------------
# Catálogo de variables (grupo y descripción legible para el tablero)
# --------------------------------------------------------------------------
_F, _C, _S, _K = "Financiero", "Conductual", "Perfil del solicitante", "Calendario"
VARIABLES_SOLICITUD = {
    "ingreso_mensual": (_F, "Ingreso mensual del solicitante"),
    "gastos_mensuales": (_F, "Gastos mensuales del solicitante"),
    "historial_mora_dias": (_F, "Días de mora en el historial"),
    "monto_solicitado": (_F, "Monto solicitado"),
    "razon_monto_ingreso": (_F, "Monto solicitado sobre ingreso mensual"),
    "capacidad_residual": (_F, "Ingreso menos gastos mensuales"),
    "cuota_estimada": (_F, "Cuota mensual estimada"),
    "razon_cuota_ingreso": (_F, "Cuota estimada sobre ingreso"),
    "razon_cuota_capacidad": (_F, "Cuota estimada sobre capacidad residual"),
    "razon_gasto_ingreso": (_F, "Gastos sobre ingreso"),
    "score_crediticio": (_F, "Score crediticio"),
    "numero_creditos_vigentes": (_F, "Número de créditos vigentes"),
    "nivel_endeudamiento": (_F, "Nivel de endeudamiento"),
    "plazo_solicitado_meses": (_F, "Plazo solicitado (meses)"),
    "tiempo_respuesta_dias": (_C, "Días que tardó el funcionario en responder"),
    "lag1_monto_func": (_C, "Monto de la solicitud anterior del funcionario"),
    "roll5_monto_func": (_C, "Monto promedio de sus 5 solicitudes anteriores"),
    "ratio_monto_vs_roll5": (_C, "Monto frente al promedio de sus 5 solicitudes anteriores"),
    "dias_desde_prev_func": (_C, "Días desde su solicitud anterior"),
    "roll10_aprob_func": (_C, "Tasa de aprobación en sus 10 solicitudes anteriores"),
    "roll10_tiempo_func": (_C, "Tiempo de respuesta en sus 10 solicitudes anteriores"),
    "roll10_excep_func": (_C, "Tasa de excepciones en sus 10 solicitudes anteriores"),
    "z_pares_tasa_aprobacion": (_C, "Tasa de aprobación frente a sus pares (Z)"),
    "z_pares_tasa_excepciones": (_C, "Tasa de excepciones frente a sus pares (Z)"),
    "z_pares_monto": (_C, "Monto promedio frente a sus pares (Z)"),
    "z_pares_razon_monto": (_C, "Razón monto/ingreso frente a sus pares (Z)"),
    "z_pares_tiempo": (_C, "Tiempo de respuesta frente a sus pares (Z)"),
    "z_monto_vs_historial_func": (_C, "Monto frente a su propio historial (Z)"),
    "edad": (_S, "Edad del solicitante"),
    "estrato_vivienda": (_S, "Estrato de la vivienda"),
    "antiguedad_laboral": (_S, "Antigüedad laboral"),
    "mes_sin": (_K, "Mes del año (seno)"), "mes_cos": (_K, "Mes del año (coseno)"),
    "dia_semana": (_K, "Día de la semana"), "trimestre": (_K, "Trimestre"),
    "fin_de_mes": (_K, "Solicitud en los últimos días del mes"), "anio": (_K, "Año de la solicitud"),
}
VARIABLES_FUNCIONARIO = {
    "z_tasa_aprobacion": (_C, "Tasa de aprobación frente a sus pares (Z)"),
    "z_tasa_excepciones": (_C, "Tasa de excepciones frente a sus pares (Z)"),
    "z_monto_promedio": (_F, "Monto promedio solicitado frente a sus pares (Z)"),
    "z_razon_monto_ingreso_prom": (_F, "Razón monto/ingreso promedio frente a sus pares (Z)"),
    "z_tiempo_resp_prom": (_C, "Tiempo de respuesta promedio frente a sus pares (Z)"),
}


# --------------------------------------------------------------------------
# Explicación con valores SHAP para Isolation Forest
# --------------------------------------------------------------------------
def _c_n(n):
    """Longitud media de camino de una búsqueda fallida en un árbol de n datos (Liu et al., 2008)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


def contribuciones_riesgo(iforest, Xt):
    """Contribución de cada variable al riesgo de cada fila (positiva = aumenta el puntaje de anomalía).

    SHAP explica la longitud media de camino que recorre cada fila en los árboles; un camino corto
    indica anomalía, por eso la contribución al riesgo es el valor SHAP con el signo cambiado.
    Se comprueba la aditividad: base + suma de contribuciones reproduce el puntaje de scikit-learn.
    """
    import shap
    ex = shap.TreeExplainer(iforest)
    sv = ex.shap_values(Xt)
    camino = np.asarray(sv).sum(axis=1) + float(np.ravel(ex.expected_value)[0])
    n_ref = min(iforest.max_samples_, 10 ** 9)
    recon = -(2.0 ** (-camino / _c_n(n_ref)))
    error = float(np.abs(recon - iforest.score_samples(Xt)).max())
    return -np.asarray(sv), error


def _top3(contrib, nombres):
    orden = np.argsort(-contrib, axis=1)[:, :3]
    var = np.array(nombres)[orden]
    val = np.take_along_axis(contrib, orden, axis=1)
    return var, val


# --------------------------------------------------------------------------
# Cálculo principal
# --------------------------------------------------------------------------
def calcular(root, fe=FE_SOLICITUD):
    """Devuelve un diccionario con las tablas del modelo dimensional y metadatos de control.

    `fe` es el nivel de variables del puntaje de solicitud (ver `modelado.NIVELES_FE`).
    """
    from src.features import desviacion_pares, perfil_funcionario, razones_solicitud

    m, etiq = M.cargar_datos(root)
    y = etiq["es_anomalia"].astype(int).values
    datos = M.agregar_temporales(m)
    idx_tr, idx_te = M.particion_prueba(len(datos), y)
    X = datos[M.columnas_entrada(fe)]

    # ---------------- nivel de solicitud ----------------
    pipe = M.armar_pipeline("isolation_forest", CONFIG_SOLICITUD, fe)
    pipe.fit(X.iloc[idx_tr].reset_index(drop=True))               # sin etiqueta
    prep, det = pipe.steps[0][1], pipe.steps[-1][1]
    Xt = prep.transform(X)
    nombres = list(prep.get_feature_names_out())
    s = det.score_anomalia(Xt)
    s_tr = np.sort(s[idx_tr])
    puntaje = 100.0 * np.searchsorted(s_tr, s, side="right") / len(s_tr)
    marcada = (s >= det.umbral_).astype(int)
    contrib, err_shap = contribuciones_riesgo(det.m_, Xt)
    var3, val3 = _top3(contrib, nombres)
    particion = np.where(np.isin(np.arange(len(datos)), idx_te), "Prueba", "Entrenamiento")

    # ---------------- nivel de funcionario ----------------
    r_all = razones_solicitud(datos)
    perfil = perfil_funcionario(datos, r_all)
    Zf = desviacion_pares(perfil, referencia=perfil)
    fdet = M.DetectorIF(contaminacion=CONTAMINACION, **CONFIG_FUNCIONARIO).fit(Zf.to_numpy(dtype=float))
    sf = fdet.score_anomalia(Zf.to_numpy(dtype=float))
    marcado_f = (sf >= fdet.umbral_).astype(int)
    puntaje_f = 100.0 * pd.Series(sf).rank(method="max", pct=True).values
    cortes = {k: float(np.quantile(sf, q)) for k, q in CORTES_NIVEL.items()}
    nivel = np.select([sf >= cortes["Crítico"], sf >= cortes["Alto"], sf >= cortes["Medio"]], ["Crítico", "Alto", "Medio"], "Bajo")
    contrib_f, err_shap_f = contribuciones_riesgo(fdet.m_, Zf.to_numpy(dtype=float))
    nom_f = list(Zf.columns)
    var3f, val3f = _top3(contrib_f, nom_f)

    # ---------------- dimensiones ----------------
    f = pd.to_datetime(datos["fecha_solicitud"])
    fechas = pd.date_range("2020-01-01", "2026-12-31", freq="D")
    meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    dim_fecha = pd.DataFrame({
        "id_fecha": fechas.strftime("%Y%m%d").astype(int), "fecha": fechas.strftime("%Y-%m-%d"),
        "anio": fechas.year, "trimestre": fechas.quarter, "mes": fechas.month,
        "nombre_mes": [meses[i - 1] for i in fechas.month], "anio_mes": fechas.strftime("%Y-%m"),
        "dia_semana": [dias[i] for i in fechas.dayofweek],
    })

    def dim(col, nombre, id_col):
        v = sorted(datos[col].astype(str).unique())
        return pd.DataFrame({id_col: range(1, len(v) + 1), nombre: v})

    dim_producto = dim("tipo_producto_financiero", "tipo_producto_financiero", "id_producto")
    dim_region = dim("region", "region", "id_region")
    dim_canal = dim("canal_solicitud", "canal_solicitud", "id_canal")
    dim_segmento = dim("segmento_cliente", "segmento_cliente", "id_segmento")
    dim_estado = dim("estado_aprobacion", "estado_aprobacion", "id_estado")
    dim_estado["es_aprobado"] = (dim_estado["estado_aprobacion"] == "Aprobado").astype(int)

    func_ids = list(perfil.index)
    dim_funcionario = pd.DataFrame({
        "id_funcionario": func_ids, "nivel_riesgo": nivel, "marcado_modelo": marcado_f,
        "orden_nivel": pd.Series(nivel).map({"Bajo": 1, "Medio": 2, "Alto": 3, "Crítico": 4}).values,
    })

    vars_all = {**{k: ("solicitud",) + v for k, v in VARIABLES_SOLICITUD.items()},
                **{f"fun:{k}": ("funcionario",) + v for k, v in VARIABLES_FUNCIONARIO.items()}}
    dim_variable = pd.DataFrame(
        [(i + 1, k.replace("fun:", ""), v[0], v[1], v[2]) for i, (k, v) in enumerate(vars_all.items())],
        columns=["id_variable", "variable", "nivel_explicacion", "grupo", "descripcion"])
    id_var = {(r.nivel_explicacion, r.variable): r.id_variable for r in dim_variable.itertuples()}
    faltan = [c for c in nombres if ("solicitud", c) not in id_var]
    assert not faltan, f"variables sin descripción: {faltan}"

    # ---------------- hechos ----------------
    mp = {col: dict(zip(d[col], d[idc])) for col, d, idc in [
        ("tipo_producto_financiero", dim_producto, "id_producto"), ("region", dim_region, "id_region"),
        ("canal_solicitud", dim_canal, "id_canal"), ("segmento_cliente", dim_segmento, "id_segmento"),
        ("estado_aprobacion", dim_estado, "id_estado")]}
    asig = lambda col: datos[col].astype(str).map(mp[col]).astype(int).values
    top_names = lambda a, nivel: [[id_var[(nivel, v)] for v in fila] for fila in a]
    t3 = np.array(top_names(var3, "solicitud"))
    fact_solicitud = pd.DataFrame({
        "id_solicitud": datos["id_solicitante"].values,
        "id_fecha": f.dt.strftime("%Y%m%d").astype(int).values,
        "id_funcionario": datos["id_funcionario_evaluador"].values,
        "id_producto": asig("tipo_producto_financiero"), "id_region": asig("region"),
        "id_canal": asig("canal_solicitud"), "id_segmento": asig("segmento_cliente"), "id_estado": asig("estado_aprobacion"),
        "monto_solicitado": datos["monto_solicitado"].values, "plazo_meses": datos["plazo_solicitado_meses"].values,
        "ingreso_mensual": datos["ingreso_mensual"].values, "razon_monto_ingreso": r_all["razon_monto_ingreso"].values,
        "razon_cuota_ingreso": datos["razon_cuota_ingreso"].values,
        "tiempo_respuesta_dias": datos["tiempo_respuesta_dias"].values,
        "aprobada": datos["aprobado_bin"].astype(int).values, "excepcion": datos["excepcion_bin"].astype(int).values,
        "puntaje_riesgo": np.round(puntaje, 2), "score_modelo": np.round(s, 6), "marcada": marcada,
        "id_variable_1": t3[:, 0], "id_variable_2": t3[:, 1], "id_variable_3": t3[:, 2],
        "contribucion_1": np.round(val3[:, 0], 5), "contribucion_2": np.round(val3[:, 1], 5), "contribucion_3": np.round(val3[:, 2], 5),
        "particion": particion,
    })

    g = datos.assign(marcada=marcada, puntaje=puntaje).groupby("id_funcionario_evaluador")
    t3f = np.array(top_names(var3f, "funcionario"))
    fact_func = pd.DataFrame({
        "id_funcionario": func_ids,
        "n_solicitudes": g.size().reindex(func_ids).values,
        "n_aprobadas": g["aprobado_bin"].sum().reindex(func_ids).astype(int).values,
        "tasa_aprobacion": g["aprobado_bin"].mean().reindex(func_ids).round(6).values,
        "tasa_excepciones": g["excepcion_bin"].mean().reindex(func_ids).round(6).values,
        "monto_promedio": g["monto_solicitado"].mean().reindex(func_ids).round(2).values,
        "razon_monto_ingreso_prom": perfil["razon_monto_ingreso_prom"].round(6).values,
        "tiempo_resp_prom": g["tiempo_respuesta_dias"].mean().reindex(func_ids).round(6).values,
        "tiempo_resp_p90": perfil["tiempo_resp_p90"].round(6).values,
        "n_marcadas": g["marcada"].sum().reindex(func_ids).astype(int).values,
        "pct_marcadas": g["marcada"].mean().reindex(func_ids).round(6).values,
        "puntaje_riesgo": np.round(puntaje_f, 2), "score_modelo": np.round(sf, 6), "marcado": marcado_f,
        "id_variable_1": t3f[:, 0], "id_variable_2": t3f[:, 1], "id_variable_3": t3f[:, 2],
        "contribucion_1": np.round(val3f[:, 0], 5), "contribucion_2": np.round(val3f[:, 1], 5), "contribucion_3": np.round(val3f[:, 2], 5),
    })
    zz = Zf.reset_index(drop=True)
    shap_f = pd.DataFrame(
        [(fid, id_var[("funcionario", v)], round(float(contrib_f[i, j]), 5), round(float(zz.iloc[i, j]), 4))
         for i, fid in enumerate(func_ids) for j, v in enumerate(nom_f)],
        columns=["id_funcionario", "id_variable", "contribucion", "valor_z"])
    sg = pd.DataFrame({"variable": nombres, "importancia_media": np.abs(contrib).mean(0), "importancia_marcadas": np.abs(contrib[marcada == 1]).mean(0)})
    sg["id_variable"] = [id_var[("solicitud", v)] for v in sg["variable"]]
    sg["pct_importancia"] = sg["importancia_media"] / sg["importancia_media"].sum()
    sg = sg.drop(columns="variable").sort_values("importancia_media", ascending=False).round(6)
    sg["posicion"] = range(1, len(sg) + 1)

    valid = pd.DataFrame({
        "id_solicitud": datos["id_solicitante"].values, "id_fecha": fact_solicitud["id_fecha"].values,
        "id_funcionario": fact_solicitud["id_funcionario"].values, "id_producto": fact_solicitud["id_producto"].values,
        "id_region": fact_solicitud["id_region"].values, "es_anomalia": y,
        "tipo_anomalia": etiq["tipo_anomalia"].astype(str).values, "marcada": marcada, "particion": particion,
    })

    tablas = {
        "dim_fecha": dim_fecha, "dim_funcionario": dim_funcionario, "dim_producto": dim_producto, "dim_region": dim_region,
        "dim_canal": dim_canal, "dim_segmento": dim_segmento, "dim_estado": dim_estado, "dim_variable": dim_variable,
        "fact_solicitud": fact_solicitud, "fact_funcionario_riesgo": fact_func, "fact_shap_funcionario": shap_f,
        "fact_shap_global": sg, "fact_validacion_sintetica": valid,
    }
    control = {"umbral_solicitud": float(det.umbral_), "umbral_funcionario": float(fdet.umbral_), "cortes_nivel": cortes,
               "error_aditividad_shap_solicitud": err_shap, "error_aditividad_shap_funcionario": err_shap_f,
               "n_entrenamiento": int(len(idx_tr)), "n_prueba": int(len(idx_te)), "variables_modelo": nombres}
    return tablas, control


# --------------------------------------------------------------------------
# Relaciones del modelo (para documentar y para comprobar integridad)
# --------------------------------------------------------------------------
RELACIONES = [
    ("fact_solicitud", "id_fecha", "dim_fecha", "id_fecha"), ("fact_solicitud", "id_funcionario", "dim_funcionario", "id_funcionario"),
    ("fact_solicitud", "id_producto", "dim_producto", "id_producto"), ("fact_solicitud", "id_region", "dim_region", "id_region"),
    ("fact_solicitud", "id_canal", "dim_canal", "id_canal"), ("fact_solicitud", "id_segmento", "dim_segmento", "id_segmento"),
    ("fact_solicitud", "id_estado", "dim_estado", "id_estado"),
    ("fact_solicitud", "id_variable_1", "dim_variable", "id_variable"),
    ("fact_funcionario_riesgo", "id_funcionario", "dim_funcionario", "id_funcionario"),
    ("fact_shap_funcionario", "id_funcionario", "dim_funcionario", "id_funcionario"),
    ("fact_shap_funcionario", "id_variable", "dim_variable", "id_variable"),
    ("fact_shap_global", "id_variable", "dim_variable", "id_variable"),
    ("fact_validacion_sintetica", "id_fecha", "dim_fecha", "id_fecha"), ("fact_validacion_sintetica", "id_funcionario", "dim_funcionario", "id_funcionario"),
    ("fact_validacion_sintetica", "id_producto", "dim_producto", "id_producto"), ("fact_validacion_sintetica", "id_region", "dim_region", "id_region"),
]

CLAVES_UNICAS = {"dim_fecha": "id_fecha", "dim_funcionario": "id_funcionario", "dim_producto": "id_producto", "dim_region": "id_region",
                 "dim_canal": "id_canal", "dim_segmento": "id_segmento", "dim_estado": "id_estado", "dim_variable": "id_variable",
                 "fact_solicitud": "id_solicitud", "fact_funcionario_riesgo": "id_funcionario", "fact_validacion_sintetica": "id_solicitud"}


def integridad(T):
    """Filas huérfanas por relación y duplicados en las claves; todo debe ser cero."""
    filas = []
    for hecho, fk, dim, pk in RELACIONES:
        huerf = int((~T[hecho][fk].isin(T[dim][pk])).sum())
        filas.append((f"Filas de {hecho}[{fk}] sin {dim}[{pk}]", huerf))
    for t, c in CLAVES_UNICAS.items():
        filas.append((f"Duplicados en {t}[{c}]", int(T[t][c].duplicated().sum())))
    return pd.DataFrame(filas, columns=["Control de integridad", "Valor"])


# --------------------------------------------------------------------------
# KPIs: definición, medida DAX y cálculo equivalente en Python
# --------------------------------------------------------------------------
def _precision_recall(T, filtro=None):
    v = T["fact_validacion_sintetica"]
    if filtro is not None:
        v = v[v["particion"] == filtro]
    vp = int(((v["marcada"] == 1) & (v["es_anomalia"] == 1)).sum())
    return vp, int(v["marcada"].sum()), int(v["es_anomalia"].sum())


def _div(a, b):
    return float(a) / float(b) if b else float("nan")


# cada KPI: id, nombre, tipo, nivel, formula (texto), dax, función(T) -> valor (a partir de las tablas exportadas)
KPIS = [
    ("K01", "Total de solicitudes", "Volumen", "Solicitud", "Número de solicitudes del contexto de filtro",
     "COUNTROWS(fact_solicitud)", lambda T: len(T["fact_solicitud"])),
    ("K02", "Monto solicitado total (COP)", "Financiero", "Solicitud", "Suma del monto solicitado",
     "SUM(fact_solicitud[monto_solicitado])", lambda T: T["fact_solicitud"]["monto_solicitado"].sum()),
    ("K03", "Monto solicitado promedio (COP)", "Financiero", "Solicitud", "Promedio del monto solicitado",
     "AVERAGE(fact_solicitud[monto_solicitado])", lambda T: T["fact_solicitud"]["monto_solicitado"].mean()),
    ("K04", "Razón monto/ingreso mediana", "Financiero", "Solicitud", "Mediana de monto solicitado / ingreso mensual",
     "MEDIAN(fact_solicitud[razon_monto_ingreso])", lambda T: T["fact_solicitud"]["razon_monto_ingreso"].median()),
    ("K05", "Tasa de aprobación", "Conductual", "Solicitud", "Solicitudes aprobadas / total de solicitudes",
     "DIVIDE(SUM(fact_solicitud[aprobada]), [Total de solicitudes])", lambda T: _div(T["fact_solicitud"]["aprobada"].sum(), len(T["fact_solicitud"]))),
    ("K06", "Tasa de excepciones", "Conductual", "Solicitud", "Solicitudes con excepción a políticas / total de solicitudes",
     "DIVIDE(SUM(fact_solicitud[excepcion]), [Total de solicitudes])", lambda T: _div(T["fact_solicitud"]["excepcion"].sum(), len(T["fact_solicitud"]))),
    ("K07", "Tiempo de respuesta promedio (días)", "Conductual", "Solicitud", "Promedio de días de respuesta",
     "AVERAGE(fact_solicitud[tiempo_respuesta_dias])", lambda T: T["fact_solicitud"]["tiempo_respuesta_dias"].mean()),
    ("K08", "Tiempo de respuesta P90 (días)", "Conductual", "Solicitud", "Percentil 90 de los días de respuesta",
     "PERCENTILE.INC(fact_solicitud[tiempo_respuesta_dias], 0.9)", lambda T: float(np.percentile(T["fact_solicitud"]["tiempo_respuesta_dias"], 90))),
    ("K09", "Solicitudes marcadas", "Riesgo", "Solicitud", "Solicitudes con puntaje en el 7,5 % superior del entrenamiento",
     "SUM(fact_solicitud[marcada])", lambda T: int(T["fact_solicitud"]["marcada"].sum())),
    ("K10", "% de solicitudes marcadas", "Riesgo", "Solicitud", "Solicitudes marcadas / total de solicitudes",
     "DIVIDE([Solicitudes marcadas], [Total de solicitudes])", lambda T: _div(T["fact_solicitud"]["marcada"].sum(), len(T["fact_solicitud"]))),
    ("K11", "Puntaje de riesgo promedio (solicitudes)", "Riesgo", "Solicitud", "Promedio del puntaje de riesgo (0 a 100)",
     "AVERAGE(fact_solicitud[puntaje_riesgo])", lambda T: T["fact_solicitud"]["puntaje_riesgo"].mean()),
    ("K12", "Funcionarios evaluados", "Volumen", "Funcionario", "Número de funcionarios del contexto de filtro",
     "COUNTROWS(fact_funcionario_riesgo)", lambda T: len(T["fact_funcionario_riesgo"])),
    ("K13", "Funcionarios en riesgo alto o crítico", "Riesgo", "Funcionario", "Funcionarios con nivel Alto o Crítico",
     "CALCULATE(COUNTROWS(fact_funcionario_riesgo), dim_funcionario[nivel_riesgo] IN {\"Alto\", \"Crítico\"})",
     lambda T: int(T["dim_funcionario"]["nivel_riesgo"].isin(["Alto", "Crítico"]).sum())),
    ("K14", "% de funcionarios en riesgo alto o crítico", "Riesgo", "Funcionario", "Funcionarios Alto o Crítico / funcionarios evaluados",
     "DIVIDE([Funcionarios en riesgo alto o crítico], [Funcionarios evaluados])",
     lambda T: _div(T["dim_funcionario"]["nivel_riesgo"].isin(["Alto", "Crítico"]).sum(), len(T["fact_funcionario_riesgo"]))),
    ("K15", "Puntaje de riesgo promedio (funcionarios)", "Riesgo", "Funcionario", "Promedio del puntaje de riesgo del funcionario (0 a 100)",
     "AVERAGE(fact_funcionario_riesgo[puntaje_riesgo])", lambda T: T["fact_funcionario_riesgo"]["puntaje_riesgo"].mean()),
    ("K16", "Contribución SHAP media", "Explicación", "Funcionario", "Promedio de la contribución de la variable al puntaje del funcionario",
     "AVERAGE(fact_shap_funcionario[contribucion])", lambda T: T["fact_shap_funcionario"]["contribucion"].mean()),
    ("K17", "Precisión de las marcas (validación sintética)", "Validación", "Solicitud", "Marcadas que son anómalas / marcadas",
     "DIVIDE(CALCULATE(COUNTROWS(fact_validacion_sintetica), fact_validacion_sintetica[marcada] = 1, fact_validacion_sintetica[es_anomalia] = 1), CALCULATE(COUNTROWS(fact_validacion_sintetica), fact_validacion_sintetica[marcada] = 1))",
     lambda T: _div(_precision_recall(T)[0], _precision_recall(T)[1])),
    ("K18", "Recall de las marcas (validación sintética)", "Validación", "Solicitud", "Anómalas que quedaron marcadas / anómalas",
     "DIVIDE(CALCULATE(COUNTROWS(fact_validacion_sintetica), fact_validacion_sintetica[marcada] = 1, fact_validacion_sintetica[es_anomalia] = 1), SUM(fact_validacion_sintetica[es_anomalia]))",
     lambda T: _div(_precision_recall(T)[0], _precision_recall(T)[2])),
]


def tabla_kpis():
    return pd.DataFrame([(k[0], k[1], k[2], k[3], k[4], k[5]) for k in KPIS],
                        columns=["Id", "Indicador", "Tipo", "Nivel", "Fórmula", "Medida DAX"])


# --------------------------------------------------------------------------
# Validación de totales: fuente frente a tablas del tablero
# --------------------------------------------------------------------------
def validacion_totales(T, root, tolerancia=1e-6, T_ref=None):
    """Compara el valor de cada control en la FUENTE (archivos del repositorio) con el de las tablas
    del tablero (leídas de los CSV). La columna `valor_power_bi` queda vacía para que el equipo la
    complete con lo que muestre Power BI; la diferencia esperada es cero."""
    raw = pd.read_csv(root / "data" / "raw" / "dataset_creditos_sinteticos.csv", encoding="utf-8-sig")
    lim = pd.read_csv(root / "data" / "processed" / "solicitudes_limpias.csv", encoding="utf-8-sig")
    mat = pd.read_csv(root / "data" / "processed" / "matriz_solicitud.csv", encoding="utf-8-sig")
    et = pd.read_csv(root / "data" / "processed" / "etiquetas_evaluacion.csv", encoding="utf-8-sig")
    perfil = pd.read_csv(root / "data" / "processed" / "perfil_funcionario.csv", encoding="utf-8-sig")
    fs, ff, fv = T["fact_solicitud"], T["fact_funcionario_riesgo"], T["fact_validacion_sintetica"]
    fecha = pd.to_datetime(lim["fecha_solicitud"])
    c = []   # (control, tabla/medida, fuente_descripcion, valor_fuente, valor_tablero)

    def add(ctrl, medida, fuente, v_fuente, v_tab):
        c.append((ctrl, medida, fuente, float(v_fuente), float(v_tab)))

    add("Registros", "fact_solicitud", "Filas de dataset_creditos_sinteticos.csv", len(raw), len(fs))
    add("Registros", "fact_funcionario_riesgo", "Funcionarios distintos en la matriz", mat["id_funcionario_evaluador"].nunique(), len(ff))
    add("Registros", "fact_validacion_sintetica", "Filas de etiquetas_evaluacion.csv", len(et), len(fv))
    add("Registros", "fact_shap_funcionario", "Funcionarios × variables del nivel funcionario", mat["id_funcionario_evaluador"].nunique() * len(VARIABLES_FUNCIONARIO), len(T["fact_shap_funcionario"]))
    add("Registros", "dim_fecha", "Días entre 2020-01-01 y 2026-12-31", len(pd.date_range("2020-01-01", "2026-12-31")), len(T["dim_fecha"]))
    add("KPI K01", "Total de solicitudes", "Filas de solicitudes_limpias.csv", len(lim), len(fs))
    add("KPI K02", "Monto solicitado total (COP)", "Suma de monto_solicitado en solicitudes_limpias.csv", lim["monto_solicitado"].sum(), fs["monto_solicitado"].sum())
    add("KPI K03", "Monto solicitado promedio (COP)", "Promedio de monto_solicitado en solicitudes_limpias.csv", lim["monto_solicitado"].mean(), fs["monto_solicitado"].mean())
    add("KPI K05", "Tasa de aprobación", "Solicitudes 'Aprobado' / total en solicitudes_limpias.csv", (lim["estado_aprobacion"] == "Aprobado").mean(), fs["aprobada"].mean())
    add("KPI K06", "Tasa de excepciones", "Solicitudes con 'Sí' en excepciones / total", (lim["excepciones_politicas_internas"] == "Sí").mean(), fs["excepcion"].mean())
    add("KPI K07", "Tiempo de respuesta promedio (días)", "Promedio en solicitudes_limpias.csv", lim["tiempo_respuesta_dias"].mean(), fs["tiempo_respuesta_dias"].mean())
    add("KPI K08", "Tiempo de respuesta P90 (días)", "Percentil 90 en solicitudes_limpias.csv", np.percentile(lim["tiempo_respuesta_dias"], 90), np.percentile(fs["tiempo_respuesta_dias"], 90))
    add("KPI K04", "Razón monto/ingreso mediana", "Mediana de monto / ingreso en matriz_solicitud.csv", mat["razon_monto_ingreso"].median(), fs["razon_monto_ingreso"].median())
    for col, dim, nom in [("id_region", "dim_region", "region"), ("id_producto", "dim_producto", "tipo_producto_financiero"),
                          ("id_canal", "dim_canal", "canal_solicitud"), ("id_estado", "dim_estado", "estado_aprobacion")]:
        mapa = dict(zip(T[dim][col], T[dim][nom]))
        for valor, n_f in fs.groupby(fs[col].map(mapa)).size().items():
            add(f"Desglose por {nom}", f"Solicitudes: {valor}", f"Conteo de {nom} = {valor} en solicitudes_limpias.csv", int((lim[nom].astype(str) == valor).sum()), n_f)
    for anio, n_f in fs.groupby(fs["id_fecha"] // 10000).size().items():
        add("Desglose por año", f"Solicitudes: {anio}", f"Conteo de solicitudes con fecha en {anio}", int((fecha.dt.year == anio).sum()), n_f)
    # indicadores por funcionario frente al perfil del repositorio
    pf = perfil.set_index("id_funcionario_evaluador")
    ffi = ff.set_index("id_funcionario")
    add("Indicadores por funcionario", "Suma de solicitudes por funcionario", "Suma de n_solicitudes en perfil_funcionario.csv", pf["n_solicitudes"].sum(), ffi["n_solicitudes"].sum())
    add("Indicadores por funcionario", "Máxima diferencia de tasa de aprobación", "perfil_funcionario.csv frente a fact_funcionario_riesgo", 0.0, (pf["tasa_aprobacion"] - ffi["tasa_aprobacion"].reindex(pf.index)).abs().max())
    add("Indicadores por funcionario", "Máxima diferencia de tiempo de respuesta promedio", "perfil_funcionario.csv frente a fact_funcionario_riesgo", 0.0, (pf["tiempo_resp_prom"] - ffi["tiempo_resp_prom"].reindex(pf.index)).abs().max())
    add("Indicadores por funcionario", "Suma de marcadas por funcionario", "Suma de marcada en fact_solicitud", fs["marcada"].sum(), ff["n_marcadas"].sum())
    # modelo: valores de las tablas frente a recálculo desde cero con el mismo código
    v = KPIS_DICT
    add("KPI K09", "Solicitudes marcadas", "Suma de marcada en fact_validacion_sintetica", fv["marcada"].sum(), v["K09"](T))
    add("KPI K13", "Funcionarios en riesgo alto o crítico", "Marcados por el modelo en fact_funcionario_riesgo", ff["marcado"].sum(), v["K13"](T))
    vp, mar, pos = _precision_recall(T)
    add("KPI K17", "Precisión de las marcas (validación sintética)", "Cálculo con etiquetas_evaluacion.csv", _div(int(((fv["marcada"] == 1) & (et["es_anomalia"].values == 1)).sum()), fv["marcada"].sum()), v["K17"](T))
    add("KPI K18", "Recall de las marcas (validación sintética)", "Cálculo con etiquetas_evaluacion.csv", _div(int(((fv["marcada"] == 1) & (et["es_anomalia"].values == 1)).sum()), et["es_anomalia"].sum()), v["K18"](T))
    # indicadores que salen del modelo (no existen en la fuente): se comparan con las tablas en memoria antes de exportar
    ref = T_ref if T_ref is not None else T
    origen = "en memoria antes de exportar" if T_ref is not None else "en las tablas cargadas"
    n_sol = len(lim)
    n_fun = mat["id_funcionario_evaluador"].nunique()
    alto = int(ref["dim_funcionario"]["nivel_riesgo"].isin(["Alto", "Crítico"]).sum())
    add("KPI K10", "% de solicitudes marcadas", f"Marcadas {origen} / filas de solicitudes_limpias.csv", ref["fact_solicitud"]["marcada"].sum() / n_sol, v["K10"](T))
    add("KPI K11", "Puntaje de riesgo promedio (solicitudes)", f"Promedio del puntaje {origen}", ref["fact_solicitud"]["puntaje_riesgo"].mean(), v["K11"](T))
    add("KPI K12", "Funcionarios evaluados", "Funcionarios distintos en matriz_solicitud.csv", n_fun, v["K12"](T))
    add("KPI K14", "% de funcionarios en riesgo alto o crítico", f"Funcionarios Alto o Crítico {origen} / funcionarios distintos en la matriz", alto / n_fun, v["K14"](T))
    add("KPI K15", "Puntaje de riesgo promedio (funcionarios)", f"Promedio del puntaje {origen}", ref["fact_funcionario_riesgo"]["puntaje_riesgo"].mean(), v["K15"](T))
    add("KPI K16", "Contribución SHAP media", f"Promedio de la contribución {origen}", ref["fact_shap_funcionario"]["contribucion"].mean(), v["K16"](T))
    out = pd.DataFrame(c, columns=["control", "medida", "fuente", "valor_fuente", "valor_tablero_python"])
    grupo = out["control"].map(lambda t: 0 if t == "Registros" else 1 if t.startswith("KPI") else 2 if t.startswith("Indicadores") else 3)
    out = out.assign(_g=grupo, _k=out["control"]).sort_values(["_g", "_k"], kind="stable").drop(columns=["_g", "_k"]).reset_index(drop=True)
    out["diferencia"] = out["valor_tablero_python"] - out["valor_fuente"]
    out["resultado"] = np.where(out["diferencia"].abs() <= tolerancia * np.maximum(1.0, out["valor_fuente"].abs()), "Coincide", "Revisar")
    out["valor_power_bi"] = ""
    return out


KPIS_DICT = {k[0]: k[6] for k in KPIS}


def exportar(T, carpeta):
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre, df in T.items():
        df.to_csv(carpeta / f"{nombre}.csv", index=False, encoding="utf-8-sig")


# --------------------------------------------------------------------------
# Guía para armar el tablero en Power BI Desktop
# --------------------------------------------------------------------------
TIPOS_DATO = {
    "id_fecha": "Número entero", "fecha": "Fecha", "anio": "Número entero", "trimestre": "Número entero", "mes": "Número entero",
    "monto_solicitado": "Número decimal", "plazo_meses": "Número entero", "ingreso_mensual": "Número decimal", "razon_monto_ingreso": "Número decimal",
    "razon_cuota_ingreso": "Número decimal", "tiempo_respuesta_dias": "Número decimal", "puntaje_riesgo": "Número decimal", "score_modelo": "Número decimal",
    "aprobada": "Número entero", "excepcion": "Número entero", "marcada": "Número entero", "marcado": "Número entero", "es_anomalia": "Número entero",
    "contribucion": "Número decimal", "valor_z": "Número decimal", "importancia_media": "Número decimal", "importancia_marcadas": "Número decimal", "pct_importancia": "Porcentaje",
}


def guia_markdown(T, control):
    L = []
    a = L.append
    a("# Guía para armar el tablero de monitoreo en Power BI Desktop\n")
    a("Esta guía acompaña al cuaderno `notebooks/03_objetivo3_tablero.ipynb`. El cuaderno deja listas las tablas del modelo dimensional (`data/dashboard/*.csv`), las medidas DAX y la hoja de validación de totales; el archivo `.pbix` lo arma el equipo con Power BI Desktop porque no se puede construir desde el entorno de análisis. Todo es un **prototipo sobre datos sintéticos**: no hay conexión con sistemas productivos ni alertas en tiempo real.\n")
    a("> **Supuestos pendientes de confirmar con el tutor y el profesor:** el puntaje del tablero usa Isolation Forest sin etiquetas con las variables del apartado 8.4.3 (ver bitácora del cuaderno, sección 10). Si se cambia el modelo, se vuelve a ejecutar el cuaderno y se recargan los CSV; las medidas no cambian.\n")
    a("## 1. Cargar las tablas\n")
    a("1. **Inicio → Obtener datos → Texto o CSV** y cargue los 13 archivos de `data/dashboard/` (codificación UTF-8). Elija **Transformar datos** y revise los tipos de dato:\n")
    a("| Columna | Tipo de dato |\n|---|---|")
    for c, t in TIPOS_DATO.items():
        a(f"| `{c}` | {t} |")
    a("\nLas columnas `id_*` y los códigos de funcionario son números enteros; el resto de columnas de texto se dejan como texto.\n")
    a("2. En `dim_fecha`: marque la tabla como **tabla de fechas** (columna `fecha`) y ordene `nombre_mes` por `mes`. En `dim_funcionario`: ordene `nivel_riesgo` por `orden_nivel`.\n")
    a("## 2. Relaciones\n")
    a("Todas son de **muchos a uno**, con dirección de filtro **única** (de la dimensión al hecho). Cree en la vista **Modelo** las relaciones de esta tabla; la del cuaderno (`tablero_relaciones.csv`) es idéntica.\n")
    a("| Tabla de hechos (lado N) | Columna | Dimensión (lado 1) | Clave |\n|---|---|---|---|")
    for h, fk, d, pk in RELACIONES:
        a(f"| `{h}` | `{fk}` | `{d}` | `{pk}` |")
    a("\n`fact_solicitud` y `fact_funcionario_riesgo` tienen además `id_variable_2` e `id_variable_3` (segunda y tercera variable que más pesan). No cree relaciones activas con ellas: use las columnas calculadas de la sección 3.\n")
    a("**Importante:** `fact_funcionario_riesgo`, `fact_shap_funcionario` y `fact_shap_global` **no** se relacionan con región, producto ni fecha. Un segmentador de región no cambia el número de funcionarios evaluados ni sus puntajes; esos indicadores se filtran con `dim_funcionario` (por ejemplo, por nivel de riesgo).\n")
    a("## 3. Columnas calculadas (texto de las variables que más pesan)\n")
    a("Cree estas columnas calculadas para mostrar el nombre legible de la variable en tablas y visuales:\n")
    a("```dax\nVariable 1 (solicitud) = LOOKUPVALUE(dim_variable[descripcion], dim_variable[id_variable], fact_solicitud[id_variable_1])\nVariable 2 (solicitud) = LOOKUPVALUE(dim_variable[descripcion], dim_variable[id_variable], fact_solicitud[id_variable_2])\nVariable 3 (solicitud) = LOOKUPVALUE(dim_variable[descripcion], dim_variable[id_variable], fact_solicitud[id_variable_3])\nVariable 1 (funcionario) = LOOKUPVALUE(dim_variable[descripcion], dim_variable[id_variable], fact_funcionario_riesgo[id_variable_1])\nVariable 2 (funcionario) = LOOKUPVALUE(dim_variable[descripcion], dim_variable[id_variable], fact_funcionario_riesgo[id_variable_2])\nVariable 3 (funcionario) = LOOKUPVALUE(dim_variable[descripcion], dim_variable[id_variable], fact_funcionario_riesgo[id_variable_3])\n```\n")
    a("## 4. Medidas DAX\n")
    a("Cree una tabla vacía para agrupar las medidas (**Introducir datos**, llámela `_Medidas`) y agregue cada una con **Nueva medida**. Los nombres deben quedar **exactamente** como aparecen aquí, porque unas medidas llaman a otras.\n")
    a("| Id | Indicador | Tipo | Fórmula |\n|---|---|---|---|")
    for k in KPIS:
        a(f"| {k[0]} | {k[1]} | {k[2]} | {k[4]} |")
    a("")
    a("```dax")
    for k in KPIS:
        a(f"{k[1]} =\n    {k[5]}\n")
    a("```\n")
    a("Los valores que debe mostrar cada medida sin filtros están en la hoja «Validación de totales» (columna «Valor en la fuente») y en `results/tables_obj3/tablero_kpis.csv`.\n")
    a("## 5. Páginas del tablero (propuesta)\n")
    a("Colores de nivel de riesgo: **Bajo** verde `#2F9E63`, **Medio** ámbar `#C99A2E`, **Alto** naranja `#D9762E`, **Crítico** rojo `#CF463D`. Use una sola escala para el resto de los visuales y no use el color como único medio para distinguir (agregue etiquetas).\n")
    a("| Página | Contenido | Datos y medidas |\n|---|---|---|")
    a("| 1. Resumen | Tarjetas con `Total de solicitudes`, `Solicitudes marcadas`, `% de solicitudes marcadas`, `Tasa de aprobación`, `Tasa de excepciones` y `Funcionarios en riesgo alto o crítico`; línea mensual de `% de solicitudes marcadas` (eje `dim_fecha[anio_mes]`); barras por región | `fact_solicitud`, `dim_fecha`, `dim_region` |")
    a("| 2. Concentración | Matriz región × tipo de producto con `% de solicitudes marcadas` y formato condicional de escala de color; segmentadores de región, producto, canal y año | `dim_region`, `dim_producto`, `dim_canal`, `dim_fecha` |")
    a("| 3. Funcionarios | Tabla con funcionario, nivel, puntaje, tasa de aprobación, tasa de excepciones, tiempo de respuesta, razón monto/ingreso y % marcadas, ordenada por puntaje; tarjetas `Funcionarios evaluados`, `Funcionarios en riesgo alto o crítico`, `% de funcionarios en riesgo alto o crítico` y `Puntaje de riesgo promedio (funcionarios)`; barras de funcionarios por nivel; segmentadores de nivel de riesgo y de funcionario | `fact_funcionario_riesgo`, `dim_funcionario` |")
    a("| 4. Explicación | Barras de `fact_shap_global[importancia_marcadas]` por `dim_variable[descripcion]` (filtro `nivel_explicacion` = solicitud) y, al elegir un funcionario, barras de `fact_shap_funcionario[contribucion]` por variable (filtro `nivel_explicacion` = funcionario); columnas `Variable 1/2/3` en la tabla | `fact_shap_global`, `fact_shap_funcionario`, `dim_variable` |")
    a("| 5. Validación sintética | `Precisión de las marcas` y `Recall de las marcas`, tabla de `tipo_anomalia` por marcada y segmentadores de partición, año, región y producto. Rotule la página: «Solo prototipo: usa la verdad conocida del dataset sintético» | `fact_validacion_sintetica` |")
    a("")
    a("En la página 4, aclare en el visual que la contribución explica **por qué el modelo marca** y no demuestra una irregularidad. Una alerta es una señal para revisar, no una prueba de fraude.\n")
    a("Las maquetas de cada página, con la posición de cada visual y de cada segmentador numerados, están en `dashboard/maquetas/` (una imagen por página y una lámina con todos los segmentadores). Son una referencia de diseño dibujada con las cifras del cuaderno; el aspecto final depende de Power BI.\n")
    a("### Segmentadores (filtros)\n")
    a("| Segmentador | Campo | Control | Páginas | Tablas de hechos que filtra |\n|---|---|---|---|---|")
    a("| Año | `dim_fecha[anio]` | Mosaico, selección múltiple | 1, 2 y 5 (sincronizado) | `fact_solicitud`, `fact_validacion_sintetica` |")
    a("| Región | `dim_region[region]` | Desplegable, selección múltiple | 1, 2 y 5 (sincronizado) | `fact_solicitud`, `fact_validacion_sintetica` |")
    a("| Tipo de producto | `dim_producto[tipo_producto_financiero]` | Desplegable, selección múltiple | 1, 2 y 5 (sincronizado) | `fact_solicitud`, `fact_validacion_sintetica` |")
    a("| Canal | `dim_canal[canal_solicitud]` | Desplegable, selección múltiple | 1 y 2 (la validación no tiene canal) | `fact_solicitud` |")
    a("| Nivel de riesgo | `dim_funcionario[nivel_riesgo]` | Mosaico con colores de riesgo | 3 | `fact_funcionario_riesgo` |")
    a("| Funcionario | `dim_funcionario[id_funcionario]` | Desplegable con búsqueda; en la página 4, selección única | 3 y 4 | `fact_funcionario_riesgo`, `fact_shap_funcionario` |")
    a("| Partición | `fact_validacion_sintetica[particion]` | Mosaico (Entrenamiento / Prueba) | 5 | `fact_validacion_sintetica` |")
    a("")
    a("Configuración: cree año, región y producto en la página 1, cópielos a las páginas 2 y 5 y use **Vista → Sincronizar segmentadores**. En **Formato → Editar interacciones**, desactive la respuesta al segmentador de región en el gráfico de barras por región (página 1) y la respuesta al segmentador de nivel en el gráfico de funcionarios por nivel (página 3). Los segmentadores de región, producto y año **no** cambian los indicadores de funcionarios (K12 a K16), porque esas tablas no tienen esos campos. Para validar los totales con la hoja de Excel, limpie todos los segmentadores.\n")
    a("## 6. Validar los totales\n")
    a("1. Abra `dashboard/Validacion_totales_tablero.xlsx` (hoja «Validación de totales»).\n2. En Power BI, sin filtros, cree una tarjeta con cada medida de la columna «Medida» y escriba el valor en la columna H.\n3. Para los desgloses (región, producto, canal, estado y año), use una tabla con la dimensión en las filas y `Total de solicitudes`.\n4. Las columnas I y J se calculan solas: la diferencia debe ser cero y el resultado «Coincide». Si algún control dice «Revisar», revise primero relaciones, tipos de dato y filtros activos.\n5. Tome una captura de la hoja ya diligenciada para el anexo del informe.\n")
    a("## 7. Capturas para el informe\n")
    a("Tome una captura de cada página (1 a 5) y de la hoja de validación diligenciada. Para cada una, el informe necesita título, fuente («Elaboración propia con datos sintéticos»), y una interpretación debajo. No copie cifras a mano: tómelas de `results/tables_obj3/tablero_*.csv`.\n")
    a("## 8. Qué verificó el cuaderno y qué no\n")
    a(f"- Verificado: las {len(RELACIONES)} relaciones no tienen filas huérfanas, las claves son únicas, y las medidas calculadas con el equivalente en Python coinciden con la fuente (hoja «Validación de totales», columnas D a G). La explicación SHAP cumple la aditividad con error del orden de {control['error_aditividad_shap_solicitud']:.0e}.")
    a("- **No verificado:** la sintaxis DAX y los visuales no se pudieron ejecutar en Power BI desde el entorno de análisis. Por eso la validación final (columnas H a J) la hace el equipo.")
    a("- El tablero muestra datos sintéticos; sus resultados no permiten concluir sobre el desempeño en una entidad real.")
    return "\n".join(L) + "\n"
