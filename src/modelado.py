"""Modelado del objetivo específico 2: batería de modelos, validación y optimización.

Todo el código de modelado del proyecto vive aquí para que el cuaderno
`02_objetivo2_modelos.ipynb` solo lo invoque y muestre resultados.

Principios que cumple este módulo (lista de chequeo técnica del ACA 2):

* La partición de prueba se separa ANTES de cualquier transformación.
* Imputación, escalado, codificación, estadísticos de pares y selección de
  variables viven dentro de un `Pipeline` y se ajustan solo con entrenamiento.
* Las etiquetas (`es_anomalia`) NO entran como variable de entrada. Se usan
  (a) para evaluar y (b) para ajustar hiperparámetros solo sobre los pliegues
  de validación del conjunto de entrenamiento. Nunca sobre el conjunto de prueba.
* Semilla fija en todas las fuentes de aleatoriedad.
* Los modelos supervisados son una referencia («techo») del enfoque no
  supervisado: se entrenan con la etiqueta del dataset sintético, que en un
  caso real no existiría.
"""
from __future__ import annotations

import time
import warnings
from functools import partial

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.cluster import DBSCAN
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

SEED = 42
COL_FUNC = "id_funcionario_evaluador"
CONTAMINACION_BASE = 0.075          # punto medio de la sensibilidad 5 % / 7,5 % / 10 %

# --------------------------------------------------------------------------
# 1. Variables
# --------------------------------------------------------------------------
# Columnas que NO entran al modelo, con la razón (se publican en una tabla).
EXCLUIDAS = {
    "id_solicitante": "Identificador: no describe comportamiento y permitiría memorizar filas.",
    "id_funcionario_evaluador": "Identificador del funcionario: solo se usa para calcular agregados dentro del Pipeline, nunca como variable.",
    "creditos_aprobados_funcionario": "Conteo que depende de la carga de cada funcionario y coincide con las aprobaciones del dataset (fuga indirecta); se reemplaza por la tasa de aprobación.",
    "fecha_solicitud": "Se descompone en variables de calendario; la fecha cruda no se usa.",
    "flag_gasto_mayor_ingreso": "Marca de calidad del ETL, redundante con capacidad_residual.",
    "flag_numerico_fuera_rango": "Marca de calidad del ETL, sin variación útil.",
    "flag_categoria_fuera_catalogo": "Marca de calidad del ETL, sin variación útil.",
    "flag_antiguedad_incoherente": "Marca de calidad del ETL, sin variación útil.",
    "flag_fecha_posterior_generacion": "Marca de calidad del ETL, sin variación útil.",
    "flag_alguna": "Marca de calidad del ETL, derivada de las anteriores.",
    "excepcion_bin": "Redundante con la categoría excepciones_politicas_internas.",
    "aprobado_bin": "Redundante con la categoría estado_aprobacion (se usa solo para agregados).",
    "decidido_bin": "Redundante con la categoría estado_aprobacion (se usa solo para agregados).",
    "mes": "Se reemplaza por su codificación cíclica (mes_sin, mes_cos).",
    "es_anomalia": "Etiqueta: solo evalúa y ajusta hiperparámetros en validación.",
    "tipo_anomalia": "Etiqueta: solo evalúa.",
}

NUM_BASE = ["edad", "ingreso_mensual", "gastos_mensuales", "estrato_vivienda", "antiguedad_laboral",
            "score_crediticio", "numero_creditos_vigentes", "nivel_endeudamiento",
            "historial_mora_dias", "monto_solicitado", "plazo_solicitado_meses", "tiempo_respuesta_dias"]
CAT_BASE = ["genero", "nivel_educativo", "estado_civil", "propiedad_vivienda", "tipo_empleo",
            "historial_crediticio", "verificacion_centrales_riesgo", "garantias_ofrecidas",
            "tipo_producto_financiero", "estado_aprobacion", "excepciones_politicas_internas",
            "canal_solicitud", "segmento_cliente", "region"]
FE_SOL = ["razon_monto_ingreso", "razon_gasto_ingreso", "capacidad_residual", "cuota_estimada",
          "razon_cuota_ingreso", "razon_cuota_capacidad"]
FE_CAL = ["mes_sin", "mes_cos", "dia_semana", "trimestre", "fin_de_mes", "anio"]
FE_ROLL = ["lag1_monto_func", "roll5_monto_func", "ratio_monto_vs_roll5", "dias_desde_prev_func",
           "roll10_aprob_func", "roll10_tiempo_func", "roll10_excep_func"]
PEER_IN = [COL_FUNC, "monto_solicitado", "razon_monto_ingreso", "tiempo_respuesta_dias",
           "aprobado_bin", "excepcion_bin"]
PEER_OUT = ["z_pares_tasa_aprobacion", "z_pares_tasa_excepciones", "z_pares_monto", "z_pares_razon_monto",
            "z_pares_tiempo", "z_monto_vs_historial_func"]
# columnas con cola pesada: se les aplica logaritmo con signo antes de escalar
LOG_COLS = ["ingreso_mensual", "gastos_mensuales", "monto_solicitado", "historial_mora_dias",
            "cuota_estimada", "capacidad_residual", "razon_monto_ingreso", "razon_cuota_capacidad",
            "razon_cuota_ingreso", "lag1_monto_func", "roll5_monto_func", "ratio_monto_vs_roll5",
            "dias_desde_prev_func"]

# Tabla de ingeniería de características (nombre, fórmula, justificación, tipo, ¿aprende parámetros?)
FE_TABLA = [
    ("razon_monto_ingreso", "monto_solicitado / ingreso_mensual", "Montos desproporcionados frente al ingreso son la señal más directa de aprobación atípica.", "Razón financiera", "No"),
    ("razon_gasto_ingreso", "gastos_mensuales / ingreso_mensual", "Holgura financiera del solicitante.", "Razón financiera", "No"),
    ("capacidad_residual", "ingreso_mensual − gastos_mensuales", "Capacidad de pago; negativa indica gastos mayores que el ingreso.", "Razón financiera", "No"),
    ("cuota_estimada", "monto_solicitado / plazo_solicitado_meses", "Pago mensual aproximado exigido (sin intereses).", "Razón financiera", "No"),
    ("razon_cuota_ingreso", "cuota_estimada / ingreso_mensual", "Carga de la cuota sobre el ingreso.", "Razón financiera", "No"),
    ("razon_cuota_capacidad", "cuota_estimada / capacidad_residual (si es positiva)", "Carga de la cuota sobre la capacidad real de pago.", "Razón financiera", "No"),
    ("mes_sin, mes_cos", "sen(2π·mes/12), cos(2π·mes/12)", "Codifican el mes de forma cíclica (diciembre queda cerca de enero) para capturar estacionalidad.", "Calendario", "No"),
    ("dia_semana, trimestre, fin_de_mes, anio", "día de la semana (0-6), trimestre (1-4), 1 si el día del mes ≥ 25, año", "Patrones de calendario: cierres de mes y de trimestre suelen concentrar excepciones.", "Calendario", "No"),
    ("lag1_monto_func", "monto de la solicitud anterior del mismo funcionario", "Rezago de orden 1: compara con lo que el funcionario acababa de aprobar.", "Rezago", "No (causal)"),
    ("roll5_monto_func", "media del monto de las 5 solicitudes previas del funcionario", "Ventana móvil: nivel reciente de montos del funcionario.", "Ventana móvil", "No (causal)"),
    ("ratio_monto_vs_roll5", "monto_solicitado / roll5_monto_func", "Salto del monto frente a su historia reciente.", "Ventana móvil", "No (causal)"),
    ("dias_desde_prev_func", "días desde la solicitud anterior del mismo funcionario", "Cambios de ritmo de trabajo del funcionario.", "Rezago", "No (causal)"),
    ("roll10_aprob_func", "tasa de aprobación de las 10 solicitudes previas del funcionario", "Racha reciente de aprobaciones.", "Ventana móvil", "No (causal)"),
    ("roll10_tiempo_func", "tiempo de respuesta medio de las 10 solicitudes previas", "Racha reciente de tiempos de respuesta.", "Ventana móvil", "No (causal)"),
    ("roll10_excep_func", "tasa de excepciones de las 10 solicitudes previas", "Racha reciente de excepciones a políticas.", "Ventana móvil", "No (causal)"),
    ("z_pares_* (5)", "(indicador del funcionario − media de pares) / desviación de pares, con indicador = tasa de aprobación, tasa de excepciones, monto promedio, razón monto/ingreso promedio y tiempo promedio", "Distancia del funcionario frente a sus compañeros; es la señal central del comportamiento atípico a nivel de funcionario.", "Desviación frente a pares", "Sí: solo con entrenamiento"),
    ("z_monto_vs_historial_func", "(monto − media del monto del funcionario) / desviación del monto del funcionario", "Cuánto se aparta esta solicitud de lo que el funcionario suele aprobar.", "Desviación frente a pares", "Sí: solo con entrenamiento"),
    ("Codificación", "One-hot de 14 variables categóricas (categorías nuevas se ignoran)", "Los modelos necesitan entradas numéricas; one-hot no impone orden entre categorías.", "Codificación", "Sí: solo con entrenamiento"),
    ("Transformación", "logaritmo con signo en 13 variables de cola pesada", "Ingreso, monto y cuotas tienen asimetría fuerte; el logaritmo evita que dominen el escalado.", "Transformación", "No"),
    ("Escalado", "StandardScaler (media 0, desviación 1) tras imputar la mediana", "Necesario para modelos lineales, KNN, SVM, red neuronal y detección por distancia.", "Escalado", "Sí: solo con entrenamiento"),
    ("Selección", "SelectPercentile con información mutua, 70 % de las variables (solo modelos supervisados)", "Descarta variables sin relación con la etiqueta; se omite en los modelos no supervisados porque requeriría la etiqueta.", "Selección", "Sí: solo con entrenamiento"),
]


def cargar_datos(root):
    """Lee la matriz de solicitud y las etiquetas (archivos separados) y las alinea."""
    m = pd.read_csv(root / "data" / "processed" / "matriz_solicitud.csv", encoding="utf-8-sig")
    e = pd.read_csv(root / "data" / "processed" / "etiquetas_evaluacion.csv", encoding="utf-8-sig")
    assert (m["id_solicitante"].values == e["id_solicitante"].values).all(), "matriz y etiquetas desalineadas"
    return m, e


def agregar_temporales(m: pd.DataFrame) -> pd.DataFrame:
    """Variables de calendario, rezagos y ventanas móviles por funcionario.

    No aprenden parámetros de los datos y son causales: la ventana de una
    solicitud solo mira solicitudes ANTERIORES del mismo funcionario
    (`shift(1)`), nunca la propia ni las posteriores. No usan la etiqueta.
    """
    d = m.copy()
    f = pd.to_datetime(d["fecha_solicitud"])
    d["_f"] = f
    d["mes_sin"] = np.sin(2 * np.pi * f.dt.month / 12)
    d["mes_cos"] = np.cos(2 * np.pi * f.dt.month / 12)
    d["dia_semana"] = f.dt.dayofweek
    d["trimestre"] = f.dt.quarter
    d["fin_de_mes"] = (f.dt.day >= 25).astype(int)
    s = d.sort_values(["_f", "id_solicitante"])
    g = s.groupby(COL_FUNC, sort=False)
    d["lag1_monto_func"] = g["monto_solicitado"].shift(1)
    d["roll5_monto_func"] = g["monto_solicitado"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    d["ratio_monto_vs_roll5"] = d["monto_solicitado"] / d["roll5_monto_func"]
    d["dias_desde_prev_func"] = (s["_f"] - g["_f"].shift(1)).dt.days
    d["roll10_aprob_func"] = g["aprobado_bin"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    d["roll10_tiempo_func"] = g["tiempo_respuesta_dias"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    d["roll10_excep_func"] = g["excepcion_bin"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    return d.drop(columns="_f")


def particion_prueba(n_filas, y, proporcion_prueba=0.30, semilla=SEED):
    """70/30 estratificada por la etiqueta. Se hace ANTES de cualquier transformación."""
    idx = np.arange(n_filas)
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(idx, test_size=proporcion_prueba, stratify=y, random_state=semilla)
    return np.sort(tr), np.sort(te)


# --------------------------------------------------------------------------
# 2. Transformadores
# --------------------------------------------------------------------------
class DesviacionPares(BaseEstimator, TransformerMixin):
    """Desviación del funcionario frente a sus pares, con estadísticos SOLO de entrenamiento.

    `fit` calcula, con las filas de entrenamiento, los indicadores de cada
    funcionario y la media y desviación del grupo. `transform` asigna a cada
    solicitud los puntajes Z de su funcionario. Funcionarios no vistos en el
    entrenamiento quedan vacíos (se imputan después). No usa la etiqueta.
    """

    def fit(self, X, y=None):
        d = pd.DataFrame(X, columns=PEER_IN)
        for c in PEER_IN[1:]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        g = d.groupby(COL_FUNC)
        ind = pd.DataFrame({
            "tasa_aprobacion": g["aprobado_bin"].mean(),
            "tasa_excepciones": g["excepcion_bin"].mean(),
            "monto_promedio": g["monto_solicitado"].mean(),
            "razon_monto": g["razon_monto_ingreso"].mean(),
            "tiempo": g["tiempo_respuesta_dias"].mean(),
        })
        self.ind_ = ind
        self.mu_ = ind.mean()
        self.sd_ = ind.std(ddof=0).replace(0, np.nan)
        self.monto_mu_ = g["monto_solicitado"].mean()
        self.monto_sd_ = g["monto_solicitado"].std(ddof=0).replace(0, np.nan)
        return self

    def transform(self, X):
        d = pd.DataFrame(X, columns=PEER_IN)
        d["monto_solicitado"] = pd.to_numeric(d["monto_solicitado"], errors="coerce")
        z = (self.ind_ - self.mu_) / self.sd_
        out = z.reindex(d[COL_FUNC].values).to_numpy(dtype=float)
        mu = d[COL_FUNC].map(self.monto_mu_).to_numpy(dtype=float)
        sd = d[COL_FUNC].map(self.monto_sd_).to_numpy(dtype=float)
        z_monto = (d["monto_solicitado"].to_numpy(dtype=float) - mu) / sd
        return np.column_stack([out, z_monto])

    def get_feature_names_out(self, input_features=None):
        return np.array(PEER_OUT)


def _slog(X):
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


NIVELES_FE = {
    "sin_fe": "Sin variables derivadas (solo las originales)",
    "8.4.3": "Solo las variables derivadas del apartado 8.4.3 (razones financieras, año y desviación frente a pares)",
    "completo": "8.4.3 más las nuevas (calendario, rezagos y ventanas móviles)",
}


def _nivel(fe):
    """Acepta False/True (compatibilidad) o el nombre del nivel."""
    if fe is False or fe == "sin_fe":
        return "sin_fe"
    if fe is True or fe == "completo":
        return "completo"
    if fe == "8.4.3":
        return "8.4.3"
    raise ValueError(fe)


def numericas_nivel(fe):
    nivel = _nivel(fe)
    if nivel == "sin_fe":
        return list(NUM_BASE)
    if nivel == "8.4.3":
        return NUM_BASE + FE_SOL + ["anio"]
    return NUM_BASE + FE_SOL + FE_CAL + FE_ROLL


def construir_preprocesador(fe=True, categoricas: bool = True) -> ColumnTransformer:
    """Imputa, aplica logaritmo con signo, escala y codifica. Se ajusta solo con entrenamiento."""
    num = numericas_nivel(fe)
    num_log = [c for c in num if c in LOG_COLS]
    num_lin = [c for c in num if c not in LOG_COLS]
    t = [
        ("num_log", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("log", FunctionTransformer(_slog, feature_names_out="one-to-one")),
                              ("esc", StandardScaler())]), num_log),
        ("num_lin", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("esc", StandardScaler())]), num_lin),
    ]
    if categoricas:
        t.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                   ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT_BASE))
    if _nivel(fe) != "sin_fe":
        t.append(("pares", Pipeline([("pares", DesviacionPares()),
                                     ("imp", SimpleImputer(strategy="median")),
                                     ("esc", StandardScaler())]), PEER_IN))
    return ColumnTransformer(t, remainder="drop", verbose_feature_names_out=False)


def columnas_entrada(fe=True):
    cols = list(numericas_nivel(fe)) + CAT_BASE
    if _nivel(fe) != "sin_fe":
        for c in PEER_IN:
            if c not in cols:
                cols.append(c)
    return cols


# --------------------------------------------------------------------------
# 3. Detectores no supervisados (misma interfaz: score_anomalia + predict)
# --------------------------------------------------------------------------
class _Detector(BaseEstimator):
    def predict(self, X):
        return (self.score_anomalia(X) >= self.umbral_).astype(int)

    def _fijar_umbral(self, X, contaminacion):
        s = self.score_anomalia(X)
        self.umbral_ = float(np.quantile(s, 1 - contaminacion))


class DetectorZ(_Detector):
    """Modelo base: se marca como atípico el registro con |Z| ≥ umbral en alguna variable."""

    def __init__(self, umbral=3.0):
        self.umbral = umbral

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mu_, self.sd_ = X.mean(0), np.where(X.std(0) == 0, 1, X.std(0))
        self.umbral_ = self.umbral
        return self

    def score_anomalia(self, X):
        return np.abs((np.asarray(X, dtype=float) - self.mu_) / self.sd_).max(axis=1)


class DetectorIF(_Detector):
    def __init__(self, n_estimators=200, max_samples=1.0, max_features=1.0, contaminacion=CONTAMINACION_BASE, random_state=SEED):
        self.n_estimators, self.max_samples, self.max_features = n_estimators, max_samples, max_features
        self.contaminacion, self.random_state = contaminacion, random_state

    def fit(self, X, y=None):
        self.m_ = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples,
                                  max_features=self.max_features, random_state=self.random_state, n_jobs=2).fit(X)
        self._fijar_umbral(X, self.contaminacion)
        return self

    def score_anomalia(self, X):
        return -self.m_.score_samples(X)


class DetectorAE(_Detector):
    """Autoencoder (red que reconstruye la entrada); el error de reconstrucción es el puntaje."""

    def __init__(self, cuello=8, alpha=1e-4, contaminacion=CONTAMINACION_BASE, random_state=SEED):
        self.cuello, self.alpha, self.contaminacion, self.random_state = cuello, alpha, contaminacion, random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        h = int(self.cuello)
        self.m_ = MLPRegressor(hidden_layer_sizes=(2 * h, h, 2 * h), alpha=self.alpha, max_iter=200,
                               random_state=self.random_state).fit(X, X)
        self._fijar_umbral(X, self.contaminacion)
        return self

    def score_anomalia(self, X):
        X = np.asarray(X, dtype=float)
        return ((self.m_.predict(X) - X) ** 2).mean(axis=1)


class DetectorDBSCAN(_Detector):
    """DBSCAN no predice: se ajusta con entrenamiento y un punto nuevo es anómalo si está
    a más de `eps` del núcleo más cercano (equivale a ser ruido)."""

    def __init__(self, eps=2.0, min_samples=10):
        self.eps, self.min_samples = eps, min_samples

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        db = DBSCAN(eps=self.eps, min_samples=int(self.min_samples), n_jobs=2).fit(X)
        nucleos = X[db.core_sample_indices_]
        self.vacio_ = len(nucleos) == 0
        if not self.vacio_:
            self.nn_ = NearestNeighbors(n_neighbors=1).fit(nucleos)
        self.umbral_ = float(np.nextafter(1.0, 2.0))
        return self

    def score_anomalia(self, X):
        X = np.asarray(X, dtype=float)
        if self.vacio_:
            return np.full(len(X), 10.0)
        return self.nn_.kneighbors(X)[0][:, 0] / self.eps


# --------------------------------------------------------------------------
# 4. Catálogo de modelos y espacios de búsqueda
# --------------------------------------------------------------------------
def _xgb(p):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                         subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
                         scale_pos_weight=p["scale_pos_weight"], eval_metric="logloss", n_jobs=2,
                         random_state=SEED, verbosity=0)


def _lgbm(p):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=p["n_estimators"], num_leaves=p["num_leaves"], learning_rate=p["learning_rate"],
                          subsample=p["subsample"], subsample_freq=1, colsample_bytree=p["colsample_bytree"],
                          min_child_samples=p["min_child_samples"], scale_pos_weight=p["scale_pos_weight"],
                          n_jobs=2, random_state=SEED, verbose=-1)


def _cat(p):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=p["iterations"], depth=p["depth"], learning_rate=p["learning_rate"],
                              l2_leaf_reg=p["l2_leaf_reg"], scale_pos_weight=p["scale_pos_weight"],
                              thread_count=2, random_seed=SEED, verbose=0, allow_writing_files=False)


_CAPAS = {"32": (32,), "64": (64,), "64-32": (64, 32)}

# espacio: parámetro -> (tipo, mínimo/opciones, máximo)
MODELOS = {
    "base_ingenuo": dict(tipo="sup", etiqueta="Base ingenuo (clase mayoritaria)", familia="Referencia",
                         build=lambda p: DummyClassifier(strategy="prior"), espacio={}),
    "logistica_l2": dict(tipo="sup", etiqueta="Regresión logística (Ridge)", familia="Lineal",
                         build=lambda p: LogisticRegression(penalty="l2", C=p["C"], class_weight="balanced", max_iter=2000, random_state=SEED),
                         espacio={"C": ("log", 1e-3, 1e2)}),
    "logistica_l1": dict(tipo="sup", etiqueta="Regresión logística (Lasso)", familia="Lineal",
                         build=lambda p: LogisticRegression(penalty="l1", solver="liblinear", C=p["C"], class_weight="balanced", max_iter=2000, random_state=SEED),
                         espacio={"C": ("log", 1e-3, 1e2)}),
    "knn": dict(tipo="sup", etiqueta="K vecinos más cercanos", familia="Distancia",
                build=lambda p: KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights=p["weights"], p=p["p"], n_jobs=2),
                espacio={"n_neighbors": ("int", 3, 51), "weights": ("cat", ["uniform", "distance"], None), "p": ("int", 1, 2)}),
    "svm": dict(tipo="sup", etiqueta="Máquina de soporte vectorial (RBF)", familia="Núcleo",
                build=lambda p: SVC(kernel="rbf", C=p["C"], gamma=p["gamma"], class_weight="balanced", random_state=SEED),
                espacio={"C": ("log", 0.1, 100.0), "gamma": ("log", 1e-4, 1e-1)}),
    "arbol": dict(tipo="sup", etiqueta="Árbol de decisión", familia="Árbol",
                  build=lambda p: DecisionTreeClassifier(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], class_weight="balanced", random_state=SEED),
                  espacio={"max_depth": ("int", 2, 12), "min_samples_leaf": ("int", 1, 30)}),
    "bosque": dict(tipo="sup", etiqueta="Bosque aleatorio", familia="Conjunto de árboles",
                   build=lambda p: RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], class_weight="balanced_subsample", n_jobs=2, random_state=SEED),
                   espacio={"n_estimators": ("int", 100, 300), "max_depth": ("int", 3, 20), "min_samples_leaf": ("int", 1, 10)}),
    "xgboost": dict(tipo="sup", etiqueta="XGBoost", familia="Potenciación de gradiente", build=_xgb,
                    espacio={"n_estimators": ("int", 100, 400), "max_depth": ("int", 2, 8), "learning_rate": ("log", 0.01, 0.3),
                             "subsample": ("float", 0.6, 1.0), "colsample_bytree": ("float", 0.5, 1.0), "scale_pos_weight": ("float", 1.0, 15.0)}),
    "lightgbm": dict(tipo="sup", etiqueta="LightGBM", familia="Potenciación de gradiente", build=_lgbm,
                     espacio={"n_estimators": ("int", 100, 400), "num_leaves": ("int", 8, 64), "learning_rate": ("log", 0.01, 0.3),
                              "subsample": ("float", 0.6, 1.0), "colsample_bytree": ("float", 0.5, 1.0),
                              "min_child_samples": ("int", 5, 50), "scale_pos_weight": ("float", 1.0, 15.0)}),
    "catboost": dict(tipo="sup", etiqueta="CatBoost", familia="Potenciación de gradiente", build=_cat,
                     espacio={"iterations": ("int", 100, 300), "depth": ("int", 3, 8), "learning_rate": ("log", 0.02, 0.3),
                              "l2_leaf_reg": ("float", 1.0, 10.0), "scale_pos_weight": ("float", 1.0, 15.0)}),
    "mlp": dict(tipo="sup", etiqueta="Red neuronal (MLP)", familia="Red neuronal",
                build=lambda p: MLPClassifier(hidden_layer_sizes=_CAPAS[p["capas"]], alpha=p["alpha"], learning_rate_init=p["learning_rate_init"],
                                              max_iter=300, early_stopping=True, random_state=SEED),
                espacio={"capas": ("cat", list(_CAPAS), None), "alpha": ("log", 1e-5, 1e-1), "learning_rate_init": ("log", 1e-3, 1e-2)}),
    # ---- no supervisados (núcleo del objetivo 2) ----
    "z_base": dict(tipo="nosup", etiqueta="Modelo base: valor Z ≥ 3", familia="Estadístico",
                   build=lambda p: DetectorZ(3.0), espacio={}),
    "isolation_forest": dict(tipo="nosup", etiqueta="Isolation Forest", familia="No supervisado",
                             build=lambda p: DetectorIF(p["n_estimators"], p["max_samples"], p["max_features"]),
                             espacio={"n_estimators": ("int", 100, 300), "max_samples": ("float", 0.2, 1.0), "max_features": ("float", 0.5, 1.0)}),
    "dbscan": dict(tipo="nosup", etiqueta="DBSCAN", familia="No supervisado",
                   build=lambda p: DetectorDBSCAN(p["eps"], p["min_samples"]),
                   espacio={"eps": ("log", 3.0, 10.0), "min_samples": ("int", 5, 50)}),
    "autoencoder": dict(tipo="nosup", etiqueta="Autoencoder", familia="No supervisado",
                        build=lambda p: DetectorAE(p["cuello"], p["alpha"]),
                        espacio={"cuello": ("int", 4, 12), "alpha": ("log", 1e-5, 1e-1)}),
}
ORDEN_SUP = ["base_ingenuo", "logistica_l2", "logistica_l1", "knn", "svm", "arbol", "bosque", "xgboost", "lightgbm", "catboost", "mlp"]
ORDEN_NOSUP = ["z_base", "isolation_forest", "dbscan", "autoencoder"]
ORDEN = ORDEN_NOSUP + ORDEN_SUP


def armar_pipeline(nombre, params, fe=True, smote=False):
    """Pipeline completo: preprocesamiento → (selección) → (SMOTE) → modelo."""
    spec = MODELOS[nombre]
    if spec["tipo"] == "nosup":
        # No supervisados: solo variables continuas (sin categorías ni selección con etiqueta)
        return Pipeline([("prep", construir_preprocesador(fe, categoricas=False)), ("modelo", spec["build"](params))])
    steps = [("prep", construir_preprocesador(fe, categoricas=True))]
    if nombre != "base_ingenuo":
        steps.append(("sel", SelectPercentile(partial(mutual_info_classif, random_state=SEED), percentile=70)))
    if smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        steps.append(("smote", SMOTE(random_state=SEED)))
        steps.append(("modelo", spec["build"](params)))
        return ImbPipeline(steps)
    steps.append(("modelo", spec["build"](params)))
    return Pipeline(steps)


def _salidas(pipe, X):
    Xt = X
    for _, paso in pipe.steps[:-1]:
        if hasattr(paso, "transform"):      # los muestreadores (SMOTE) solo actúan al ajustar
            Xt = paso.transform(Xt)
    est = pipe.steps[-1][1]
    pred = est.predict(Xt)
    if hasattr(est, "score_anomalia"):
        s = est.score_anomalia(Xt)
    elif hasattr(est, "predict_proba"):
        s = est.predict_proba(Xt)[:, 1]
    else:
        s = est.decision_function(Xt)
    return np.asarray(pred).astype(int), np.asarray(s, dtype=float)


def metricas(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "auc_roc": roc_auc_score(y, score),
        "auc_pr": average_precision_score(y, score),
        "mcc": matthews_corrcoef(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1_anomalia": f1_score(y, pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def ajustar_y_evaluar(nombre, params, X_tr, y_tr, X_ev, y_ev, fe=True, smote=False):
    """Ajusta con (X_tr, y_tr) y evalúa en (X_ev, y_ev). Devuelve métricas, tiempo y salidas."""
    pipe = armar_pipeline(nombre, params, fe, smote)
    t0 = time.perf_counter()
    if MODELOS[nombre]["tipo"] == "nosup":
        pipe.fit(X_tr)                      # no supervisado: no ve la etiqueta
    else:
        pipe.fit(X_tr, y_tr)
    t_fit = time.perf_counter() - t0
    pred, s = _salidas(pipe, X_ev)
    m = metricas(y_ev, pred, s)
    m["tiempo_ajuste_s"] = t_fit
    return m, pred, s, pipe


# --------------------------------------------------------------------------
# 5. Optimización con Optuna
# --------------------------------------------------------------------------
def tabla_espacios():
    filas = []
    for n in ORDEN:
        for p, (tipo, a, b) in MODELOS[n]["espacio"].items():
            if tipo == "cat":
                rango = ", ".join(map(str, a)); t = "categórico"
            else:
                rango = f"{a:g} a {b:g}"; t = {"int": "entero", "float": "real", "log": "real (escala log)"}[tipo]
            filas.append((MODELOS[n]["etiqueta"], p, t, rango))
        if not MODELOS[n]["espacio"]:
            filas.append((MODELOS[n]["etiqueta"], "(sin hiperparámetros)", "—", "—"))
    return pd.DataFrame(filas, columns=["Modelo", "Hiperparámetro", "Tipo", "Rango de búsqueda"])


def _sugerir(trial, espacio):
    p = {}
    for k, (tipo, a, b) in espacio.items():
        if tipo == "int":
            p[k] = trial.suggest_int(k, a, b)
        elif tipo == "float":
            p[k] = trial.suggest_float(k, a, b)
        elif tipo == "log":
            p[k] = trial.suggest_float(k, a, b, log=True)
        else:
            p[k] = trial.suggest_categorical(k, a)
    return p


def optimizar(nombre, X, y, n_trials=20, fe=True, semilla=SEED, log=None):
    """Optuna sobre pliegues de validación INTERNOS del entrenamiento (3 pliegues).

    Mismo número de pruebas, mismo muestreador (TPE) y mismos pliegues para todos
    los modelos. Devuelve los mejores parámetros, el mejor F1 macro y la tabla de pruebas.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    spec = MODELOS[nombre]
    if not spec["espacio"]:
        m, *_ = ajustar_y_evaluar(nombre, {}, X, y, X, y, fe)
        return {}, np.nan, pd.DataFrame(), 0.0
    cv = StratifiedKFold(3, shuffle=True, random_state=semilla + 1)
    pliegues = list(cv.split(X, y))
    t0 = time.perf_counter()

    def objetivo(trial):
        p = _sugerir(trial, spec["espacio"])
        vals = []
        for tr, va in pliegues:
            try:
                m, *_ = ajustar_y_evaluar(nombre, p, X.iloc[tr], y[tr], X.iloc[va], y[va], fe)
                vals.append(m["f1_macro"])
            except Exception:
                vals.append(0.0)
        return float(np.mean(vals))

    estudio = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=semilla))
    estudio.optimize(objetivo, n_trials=n_trials)
    dur = time.perf_counter() - t0
    pruebas = estudio.trials_dataframe()[["number", "value"] + [c for c in estudio.trials_dataframe().columns if c.startswith("params_")]]
    pruebas.insert(0, "modelo", nombre)
    if log:
        log(f"  {nombre}: mejor F1 macro interno {estudio.best_value:.4f} ({dur:.0f} s)")
    return estudio.best_params, float(estudio.best_value), pruebas, dur


# --------------------------------------------------------------------------
# 6. Validación repetida
# --------------------------------------------------------------------------
def validacion_repetida(nombre, params, X, y, fe=True, smote=False, n_splits=5, n_repeats=2, semilla=SEED):
    """RepeatedStratifiedKFold 5×2 sobre el ENTRENAMIENTO. Devuelve una fila por partición."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=semilla)
    filas = []
    for i, (tr, va) in enumerate(rskf.split(X, y)):
        m, *_ = ajustar_y_evaluar(nombre, params, X.iloc[tr], y[tr], X.iloc[va], y[va], fe, smote)
        m.update({"modelo": nombre, "particion": i + 1, "repeticion": i // n_splits + 1, "pliegue": i % n_splits + 1})
        filas.append(m)
    return pd.DataFrame(filas)


def resumir(df, columnas=("f1_macro", "f1_anomalia", "auc_roc", "auc_pr", "mcc", "precision", "recall", "tiempo_ajuste_s")):
    g = df.groupby("modelo", sort=False)
    out = {}
    for c in columnas:
        out[c + "_media"] = g[c].mean()
        out[c + "_sd"] = g[c].std(ddof=1)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# 7. Nivel de funcionario
# --------------------------------------------------------------------------
def z_funcionarios(datos_entrenamiento: pd.DataFrame, todos: pd.DataFrame):
    """Perfiles de los 220 funcionarios con la referencia de pares calculada SOLO con entrenamiento."""
    from src.features import desviacion_pares, perfil_funcionario, razones_solicitud
    r_tr = razones_solicitud(datos_entrenamiento)
    perfil_tr = perfil_funcionario(datos_entrenamiento, r_tr)
    r_all = razones_solicitud(todos)
    perfil_all = perfil_funcionario(todos, r_all)
    # el perfil de cada funcionario se calcula con sus solicitudes de entrenamiento (protocolo),
    # y se completa con el perfil completo solo si un funcionario no tuviera solicitudes de entrenamiento
    perfil = perfil_tr.reindex(perfil_all.index).fillna(perfil_all)
    z = desviacion_pares(perfil, referencia=perfil_tr)
    return perfil, z


def evaluar_funcionario(nombre, params, Z, y, contaminacion=CONTAMINACION_BASE):
    """Ajusta un detector no supervisado sobre los puntajes Z de los 220 funcionarios y evalúa."""
    X = Z.to_numpy(dtype=float)
    if nombre == "z_base":
        det = DetectorZ(3.0)
    elif nombre == "isolation_forest":
        det = DetectorIF(contaminacion=contaminacion, **params)
    elif nombre == "autoencoder":
        det = DetectorAE(contaminacion=contaminacion, **params)
    elif nombre == "dbscan":
        det = DetectorDBSCAN(**params)
    det.fit(X)
    return det.predict(X).astype(int), det.score_anomalia(X)


def _detector_funcionario(nombre, params, contaminacion):
    if nombre == "z_base":
        return DetectorZ(3.0)
    if nombre == "isolation_forest":
        return DetectorIF(contaminacion=contaminacion, **params)
    if nombre == "autoencoder":
        return DetectorAE(contaminacion=contaminacion, **params)
    if nombre == "dbscan":
        return DetectorDBSCAN(**params)
    raise ValueError(nombre)


def evaluar_funcionario_oof(nombre, params, perfil, y, contaminacion=CONTAMINACION_BASE, n_splits=3, n_repeats=20, semilla=SEED):
    """Evaluación fuera de muestra (out-of-fold) a nivel de funcionario.

    Los funcionarios se dividen en pliegues estratificados por la etiqueta. En cada pliegue, la referencia
    de pares y el detector se calculan SOLO con los funcionarios de los otros pliegues, y se puntúa a los
    del pliegue retenido: ningún funcionario se evalúa con un modelo que lo haya visto. La etiqueta solo
    se usa para estratificar y para medir. Devuelve una fila de métricas por repetición y las
    predicciones de la primera repetición (para el intervalo de remuestreo).
    """
    from src.features import desviacion_pares
    y = np.asarray(y).astype(int)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=semilla)
    pred, sc, filas, primera = np.zeros(len(y), int), np.zeros(len(y)), [], None
    for i, (tr, va) in enumerate(rskf.split(np.zeros(len(y)), y)):
        ref = perfil.iloc[tr]
        Ztr = desviacion_pares(ref, referencia=ref).fillna(0.0).to_numpy(float)
        Zva = desviacion_pares(perfil.iloc[va], referencia=ref).fillna(0.0).to_numpy(float)
        det = _detector_funcionario(nombre, params, contaminacion).fit(Ztr)
        pred[va], sc[va] = det.predict(Zva), det.score_anomalia(Zva)
        if i % n_splits == n_splits - 1:
            mt = metricas(y, pred, sc)
            filas.append({"repeticion": i // n_splits + 1, "marcados": int(pred.sum()), "tp": mt["tp"], "precision": mt["precision"],
                          "recall": mt["recall"], "f1_anomalia": mt["f1_anomalia"]})
            if primera is None:
                primera = pred.copy()
    return pd.DataFrame(filas), primera


# --------------------------------------------------------------------------
# 8. Utilidades adicionales (sensibilidad y remuestreo)
# --------------------------------------------------------------------------
def validacion_detector(fabrica, X, y, fe=True, n_splits=5, n_repeats=2, semilla=SEED):
    """Validación repetida 5×2 de un detector no supervisado construido por `fabrica()`.

    Sirve para el análisis de sensibilidad (por ejemplo, otro valor de contaminación).
    El detector se ajusta sin etiquetas; la etiqueta solo evalúa el pliegue de validación.
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=semilla)
    filas = []
    for i, (tr, va) in enumerate(rskf.split(X, y)):
        pipe = Pipeline([("prep", construir_preprocesador(fe, categoricas=False)), ("modelo", fabrica())])
        pipe.fit(X.iloc[tr])
        pred, s = _salidas(pipe, X.iloc[va])
        m = metricas(y[va], pred, s)
        m["particion"] = i + 1
        filas.append(m)
    return pd.DataFrame(filas)


def ic_bootstrap(y, pred, n=1000, semilla=SEED, nivel=0.95):
    """Intervalo percentil del F1 de la clase anómala por remuestreo con reemplazo."""
    rng = np.random.default_rng(semilla)
    y, pred = np.asarray(y), np.asarray(pred)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(f1_score(y[i], pred[i], zero_division=0))
    a = (1 - nivel) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


# --------------------------------------------------------------------------
# 9. Desglose por tipo de anomalía, segmentos y réplicas del generador
# --------------------------------------------------------------------------
TIPOS = ["Monto desproporcionado vs. ingreso", "Funcionario con aprobación atípica", "Tiempo de respuesta excesivo"]


def metricas_por_clase(y, pred):
    """Precision, recall y F1 de cada clase (normal = 0, anómala = 1)."""
    filas = []
    for c, nombre in ((0, "Normal"), (1, "Anómala")):
        filas.append({"clase": nombre, "precision": precision_score(y, pred, pos_label=c, zero_division=0),
                      "recall": recall_score(y, pred, pos_label=c, zero_division=0),
                      "f1": f1_score(y, pred, pos_label=c, zero_division=0), "soporte": int((np.asarray(y) == c).sum())})
    return filas


def metricas_por_tipo(y, tipo, pred):
    """Desempeño por tipo de anomalía.

    Para cada tipo se evalúa sobre las solicitudes normales más las anómalas de ese tipo:
    recall = anómalas del tipo detectadas / anómalas del tipo; precision = detectadas del tipo /
    (detectadas del tipo + falsas alarmas entre las normales); F1 con ambas.
    """
    y, tipo, pred = np.asarray(y), np.asarray(tipo), np.asarray(pred)
    normal = y == 0
    filas = []
    for t in TIPOS:
        m = tipo == t
        tp = int(((pred == 1) & m).sum()); fn = int(((pred == 0) & m).sum()); fp = int(((pred == 1) & normal).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        filas.append({"tipo": t, "n_tipo": int(m.sum()), "detectadas": tp, "precision": prec, "recall": rec, "f1": f1})
    return filas


def predicciones_oof(nombre, params, X, y, fe=True, n_splits=5, semilla=SEED):
    """Predicciones fuera de muestra sobre TODAS las filas (5 pliegues estratificados).

    Se usa solo en análisis exploratorios por subgrupo y por año con modelos no supervisados,
    cuyo ajuste no ve la etiqueta; así cada fila se predice con un modelo que no la incluyó.
    """
    pred = np.zeros(len(y), dtype=int); score = np.zeros(len(y))
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=semilla).split(X, y):
        _, p, s, _ = ajustar_y_evaluar(nombre, params, X.iloc[tr].reset_index(drop=True), y[tr], X.iloc[va].reset_index(drop=True), y[va], fe)
        pred[va], score[va] = p, s
    return pred, score


def preparar_matriz(raw: pd.DataFrame) -> pd.DataFrame:
    """Matriz de solicitud a partir del archivo original (mismas columnas que el cuaderno 01).

    El ETL del cuaderno 01 no cambia valores (solo tipos y banderas), por lo que para las réplicas
    basta quitar las etiquetas y agregar las razones financieras.
    """
    from src.features import razones_solicitud
    d = raw.drop(columns=["es_anomalia", "tipo_anomalia"])
    r = razones_solicitud(d)
    return pd.concat([d, r.drop(columns=["mes", "anio"]).join(r[["mes", "anio"]])], axis=1)


def generar_replica(semilla, carpeta, root):
    """Ejecuta el generador con otra semilla SIN modificar src/generar_dataset.py.

    Se copia el script a `carpeta`, se cambia la única línea `SEED = 42` y se ejecuta allí.
    Con semilla 42 reproduce el archivo original (se comprueba en el cuaderno).
    """
    import pathlib, subprocess, sys
    fuente = (pathlib.Path(root) / "src" / "generar_dataset.py").read_text(encoding="utf-8")
    assert fuente.count("SEED = 42") == 1, "El generador cambió: revisar la sustitución de la semilla."
    carpeta = pathlib.Path(carpeta); carpeta.mkdir(parents=True, exist_ok=True)
    script = carpeta / f"generador_semilla_{semilla}.py"
    script.write_text(fuente.replace("SEED = 42", f"SEED = {int(semilla)}"), encoding="utf-8")
    subprocess.run([sys.executable, str(script), "--salida", str(carpeta / f"s{semilla}")], check=True, capture_output=True)
    return pd.read_csv(carpeta / f"s{semilla}" / "dataset_creditos_sinteticos.csv", encoding="utf-8-sig")


NUCLEO = ["z_base", "isolation_forest", "dbscan", "autoencoder"]
HIPER_FUNCIONARIO = {"z_base": {}, "isolation_forest": dict(n_estimators=200, max_samples=1.0, max_features=1.0),
                     "autoencoder": dict(cuello=3, alpha=1e-3), "dbscan": dict(eps=1.5, min_samples=5)}


def evaluar_replica(semilla, hiper, carpeta, root):
    """Una réplica completa: genera, prepara, parte 70/30 y evalúa los cuatro modelos del núcleo.

    Usa los hiperparámetros ya ajustados con el dataset principal (no se vuelve a optimizar),
    y devuelve el F1 de la clase anómala en la prueba a nivel de solicitud y a nivel de funcionario.
    """
    raw = generar_replica(semilla, carpeta, root)
    y = raw["es_anomalia"].astype(int).values
    d = agregar_temporales(preparar_matriz(raw))
    X = d[columnas_entrada(True)]
    tr, te = particion_prueba(len(d), y)
    fila = {"semilla": semilla, "prevalencia": float(y.mean())}
    for nom in NUCLEO:
        m, *_ = ajustar_y_evaluar(nom, hiper[nom], X.iloc[tr].reset_index(drop=True), y[tr], X.iloc[te].reset_index(drop=True), y[te], True)
        fila[f"sol_{nom}"] = m["f1_anomalia"]; fila[f"sol_auc_pr_{nom}"] = m["auc_pr"]
    # nivel de funcionario: perfiles con las solicitudes de entrenamiento; se evalúa sobre todos los funcionarios
    yf = (raw.assign(f=(raw["tipo_anomalia"] == "Funcionario con aprobación atípica").astype(int))
          .groupby(COL_FUNC)["f"].max())
    perfil, Zf = z_funcionarios(d.iloc[tr], d)
    yf = yf.reindex(Zf.index).astype(int).values; Zf = Zf.fillna(0.0)
    fila["n_funcionarios_atipicos"] = int(yf.sum())
    for nom in NUCLEO:
        pred, sc = evaluar_funcionario(nom, HIPER_FUNCIONARIO[nom], Zf, yf)
        fila[f"fun_{nom}"] = f1_score(yf, pred, zero_division=0)
    return fila
