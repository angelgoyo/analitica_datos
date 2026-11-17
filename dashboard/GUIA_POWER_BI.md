# Guía para armar el tablero de monitoreo en Power BI Desktop

Esta guía acompaña al cuaderno `notebooks/03_objetivo3_tablero.ipynb`. El cuaderno deja listas las tablas del modelo dimensional (`data/dashboard/*.csv`), las medidas DAX y la hoja de validación de totales; el archivo `.pbix` lo arma el equipo con Power BI Desktop porque no se puede construir desde el entorno de análisis. Todo es un **prototipo sobre datos sintéticos**: no hay conexión con sistemas productivos ni alertas en tiempo real.

> **Supuestos pendientes de confirmar con el tutor y el profesor:** el puntaje del tablero usa Isolation Forest sin etiquetas con las variables del apartado 8.4.3 (ver bitácora del cuaderno, sección 10). Si se cambia el modelo, se vuelve a ejecutar el cuaderno y se recargan los CSV; las medidas no cambian.

## 1. Cargar las tablas

1. **Inicio → Obtener datos → Texto o CSV** y cargue los 13 archivos de `data/dashboard/` (codificación UTF-8). Elija **Transformar datos** y revise los tipos de dato:

| Columna | Tipo de dato |
|---|---|
| `id_fecha` | Número entero |
| `fecha` | Fecha |
| `anio` | Número entero |
| `trimestre` | Número entero |
| `mes` | Número entero |
| `monto_solicitado` | Número decimal |
| `plazo_meses` | Número entero |
| `ingreso_mensual` | Número decimal |
| `razon_monto_ingreso` | Número decimal |
| `razon_cuota_ingreso` | Número decimal |
| `tiempo_respuesta_dias` | Número decimal |
| `puntaje_riesgo` | Número decimal |
| `score_modelo` | Número decimal |
| `aprobada` | Número entero |
| `excepcion` | Número entero |
| `marcada` | Número entero |
| `marcado` | Número entero |
| `es_anomalia` | Número entero |
| `contribucion` | Número decimal |
| `valor_z` | Número decimal |
| `importancia_media` | Número decimal |
| `importancia_marcadas` | Número decimal |
| `pct_importancia` | Porcentaje |

Las columnas `id_*` y los códigos de funcionario son números enteros; el resto de columnas de texto se dejan como texto.

2. En `dim_fecha`: marque la tabla como **tabla de fechas** (columna `fecha`) y ordene `nombre_mes` por `mes`. En `dim_funcionario`: ordene `nivel_riesgo` por `orden_nivel`.

## 2. Relaciones

Todas son de **muchos a uno**, con dirección de filtro **única** (de la dimensión al hecho). Cree en la vista **Modelo** las relaciones de esta tabla; la del cuaderno (`tablero_relaciones.csv`) es idéntica.

| Tabla de hechos (lado N) | Columna | Dimensión (lado 1) | Clave |
|---|---|---|---|
| `fact_solicitud` | `id_fecha` | `dim_fecha` | `id_fecha` |
| `fact_solicitud` | `id_funcionario` | `dim_funcionario` | `id_funcionario` |
| `fact_solicitud` | `id_producto` | `dim_producto` | `id_producto` |
| `fact_solicitud` | `id_region` | `dim_region` | `id_region` |
| `fact_solicitud` | `id_canal` | `dim_canal` | `id_canal` |
| `fact_solicitud` | `id_segmento` | `dim_segmento` | `id_segmento` |
| `fact_solicitud` | `id_estado` | `dim_estado` | `id_estado` |
| `fact_solicitud` | `id_variable_1` | `dim_variable` | `id_variable` |
| `fact_funcionario_riesgo` | `id_funcionario` | `dim_funcionario` | `id_funcionario` |
| `fact_shap_funcionario` | `id_funcionario` | `dim_funcionario` | `id_funcionario` |
| `fact_shap_funcionario` | `id_variable` | `dim_variable` | `id_variable` |
| `fact_shap_global` | `id_variable` | `dim_variable` | `id_variable` |
| `fact_validacion_sintetica` | `id_fecha` | `dim_fecha` | `id_fecha` |
| `fact_validacion_sintetica` | `id_funcionario` | `dim_funcionario` | `id_funcionario` |
| `fact_validacion_sintetica` | `id_producto` | `dim_producto` | `id_producto` |
| `fact_validacion_sintetica` | `id_region` | `dim_region` | `id_region` |

`fact_solicitud` y `fact_funcionario_riesgo` tienen además `id_variable_2` e `id_variable_3` (segunda y tercera variable que más pesan). No cree relaciones activas con ellas: use las columnas calculadas de la sección 3.

**Importante:** `fact_funcionario_riesgo`, `fact_shap_funcionario` y `fact_shap_global` **no** se relacionan con región, producto ni fecha. Un segmentador de región no cambia el número de funcionarios evaluados ni sus puntajes; esos indicadores se filtran con `dim_funcionario` (por ejemplo, por nivel de riesgo).

## 3. Columnas calculadas (texto de las variables que más pesan)

Cree estas columnas calculadas para mostrar el nombre legible de la variable en tablas y visuales:

```dax
Variable 1 (solicitud) = LOOKUPVALUE(dim_variable[descripcion], dim_variable[id_variable], fact_solicitud[id_variable_1])
Variable 2 (solicitud) = LOOKUPVALUE(dim_variable[descripcion], dim_variable[id_variable], fact_solicitud[id_variable_2])
Variable 3 (solicitud) = LOOKUPVALUE(dim_variable[descripcion], dim_variable[id_variable], fact_solicitud[id_variable_3])
Variable 1 (funcionario) = LOOKUPVALUE(dim_variable[descripcion], dim_variable[id_variable], fact_funcionario_riesgo[id_variable_1])
Variable 2 (funcionario) = LOOKUPVALUE(dim_variable[descripcion], dim_variable[id_variable], fact_funcionario_riesgo[id_variable_2])
Variable 3 (funcionario) = LOOKUPVALUE(dim_variable[descripcion], dim_variable[id_variable], fact_funcionario_riesgo[id_variable_3])
```

## 4. Medidas DAX

Cree una tabla vacía para agrupar las medidas (**Introducir datos**, llámela `_Medidas`) y agregue cada una con **Nueva medida**. Los nombres deben quedar **exactamente** como aparecen aquí, porque unas medidas llaman a otras.

| Id | Indicador | Tipo | Fórmula |
|---|---|---|---|
| K01 | Total de solicitudes | Volumen | Número de solicitudes del contexto de filtro |
| K02 | Monto solicitado total (COP) | Financiero | Suma del monto solicitado |
| K03 | Monto solicitado promedio (COP) | Financiero | Promedio del monto solicitado |
| K04 | Razón monto/ingreso mediana | Financiero | Mediana de monto solicitado / ingreso mensual |
| K05 | Tasa de aprobación | Conductual | Solicitudes aprobadas / total de solicitudes |
| K06 | Tasa de excepciones | Conductual | Solicitudes con excepción a políticas / total de solicitudes |
| K07 | Tiempo de respuesta promedio (días) | Conductual | Promedio de días de respuesta |
| K08 | Tiempo de respuesta P90 (días) | Conductual | Percentil 90 de los días de respuesta |
| K09 | Solicitudes marcadas | Riesgo | Solicitudes con puntaje en el 7,5 % superior del entrenamiento |
| K10 | % de solicitudes marcadas | Riesgo | Solicitudes marcadas / total de solicitudes |
| K11 | Puntaje de riesgo promedio (solicitudes) | Riesgo | Promedio del puntaje de riesgo (0 a 100) |
| K12 | Funcionarios evaluados | Volumen | Número de funcionarios del contexto de filtro |
| K13 | Funcionarios en riesgo alto o crítico | Riesgo | Funcionarios con nivel Alto o Crítico |
| K14 | % de funcionarios en riesgo alto o crítico | Riesgo | Funcionarios Alto o Crítico / funcionarios evaluados |
| K15 | Puntaje de riesgo promedio (funcionarios) | Riesgo | Promedio del puntaje de riesgo del funcionario (0 a 100) |
| K16 | Contribución SHAP media | Explicación | Promedio de la contribución de la variable al puntaje del funcionario |
| K17 | Precisión de las marcas (validación sintética) | Validación | Marcadas que son anómalas / marcadas |
| K18 | Recall de las marcas (validación sintética) | Validación | Anómalas que quedaron marcadas / anómalas |

```dax
Total de solicitudes =
    COUNTROWS(fact_solicitud)

Monto solicitado total (COP) =
    SUM(fact_solicitud[monto_solicitado])

Monto solicitado promedio (COP) =
    AVERAGE(fact_solicitud[monto_solicitado])

Razón monto/ingreso mediana =
    MEDIAN(fact_solicitud[razon_monto_ingreso])

Tasa de aprobación =
    DIVIDE(SUM(fact_solicitud[aprobada]), [Total de solicitudes])

Tasa de excepciones =
    DIVIDE(SUM(fact_solicitud[excepcion]), [Total de solicitudes])

Tiempo de respuesta promedio (días) =
    AVERAGE(fact_solicitud[tiempo_respuesta_dias])

Tiempo de respuesta P90 (días) =
    PERCENTILE.INC(fact_solicitud[tiempo_respuesta_dias], 0.9)

Solicitudes marcadas =
    SUM(fact_solicitud[marcada])

% de solicitudes marcadas =
    DIVIDE([Solicitudes marcadas], [Total de solicitudes])

Puntaje de riesgo promedio (solicitudes) =
    AVERAGE(fact_solicitud[puntaje_riesgo])

Funcionarios evaluados =
    COUNTROWS(fact_funcionario_riesgo)

Funcionarios en riesgo alto o crítico =
    CALCULATE(COUNTROWS(fact_funcionario_riesgo), dim_funcionario[nivel_riesgo] IN {"Alto", "Crítico"})

% de funcionarios en riesgo alto o crítico =
    DIVIDE([Funcionarios en riesgo alto o crítico], [Funcionarios evaluados])

Puntaje de riesgo promedio (funcionarios) =
    AVERAGE(fact_funcionario_riesgo[puntaje_riesgo])

Contribución SHAP media =
    AVERAGE(fact_shap_funcionario[contribucion])

Precisión de las marcas (validación sintética) =
    DIVIDE(CALCULATE(COUNTROWS(fact_validacion_sintetica), fact_validacion_sintetica[marcada] = 1, fact_validacion_sintetica[es_anomalia] = 1), CALCULATE(COUNTROWS(fact_validacion_sintetica), fact_validacion_sintetica[marcada] = 1))

Recall de las marcas (validación sintética) =
    DIVIDE(CALCULATE(COUNTROWS(fact_validacion_sintetica), fact_validacion_sintetica[marcada] = 1, fact_validacion_sintetica[es_anomalia] = 1), SUM(fact_validacion_sintetica[es_anomalia]))

```

Los valores que debe mostrar cada medida sin filtros están en la hoja «Validación de totales» (columna «Valor en la fuente») y en `results/tables_obj3/tablero_kpis.csv`.

## 5. Páginas del tablero (propuesta)

Colores de nivel de riesgo: **Bajo** verde `#2F9E63`, **Medio** ámbar `#C99A2E`, **Alto** naranja `#D9762E`, **Crítico** rojo `#CF463D`. Use una sola escala para el resto de los visuales y no use el color como único medio para distinguir (agregue etiquetas).

| Página | Contenido | Datos y medidas |
|---|---|---|
| 1. Resumen | Tarjetas con `Total de solicitudes`, `Solicitudes marcadas`, `% de solicitudes marcadas`, `Tasa de aprobación`, `Tasa de excepciones` y `Funcionarios en riesgo alto o crítico`; línea mensual de `% de solicitudes marcadas` (eje `dim_fecha[anio_mes]`); barras por región | `fact_solicitud`, `dim_fecha`, `dim_region` |
| 2. Concentración | Matriz región × tipo de producto con `% de solicitudes marcadas` y formato condicional de escala de color; segmentadores de región, producto, canal y año | `dim_region`, `dim_producto`, `dim_canal`, `dim_fecha` |
| 3. Funcionarios | Tabla con funcionario, nivel, puntaje, tasa de aprobación, tasa de excepciones, tiempo de respuesta, razón monto/ingreso y % marcadas, ordenada por puntaje; tarjetas `Funcionarios evaluados`, `Funcionarios en riesgo alto o crítico`, `% de funcionarios en riesgo alto o crítico` y `Puntaje de riesgo promedio (funcionarios)`; barras de funcionarios por nivel; segmentadores de nivel de riesgo y de funcionario | `fact_funcionario_riesgo`, `dim_funcionario` |
| 4. Explicación | Barras de `fact_shap_global[importancia_marcadas]` por `dim_variable[descripcion]` (filtro `nivel_explicacion` = solicitud) y, al elegir un funcionario, barras de `fact_shap_funcionario[contribucion]` por variable (filtro `nivel_explicacion` = funcionario); columnas `Variable 1/2/3` en la tabla | `fact_shap_global`, `fact_shap_funcionario`, `dim_variable` |
| 5. Validación sintética | `Precisión de las marcas` y `Recall de las marcas`, tabla de `tipo_anomalia` por marcada y segmentadores de partición, año, región y producto. Rotule la página: «Solo prototipo: usa la verdad conocida del dataset sintético» | `fact_validacion_sintetica` |

En la página 4, aclare en el visual que la contribución explica **por qué el modelo marca** y no demuestra una irregularidad. Una alerta es una señal para revisar, no una prueba de fraude.

Las maquetas de cada página, con la posición de cada visual y de cada segmentador numerados, están en `dashboard/maquetas/` (una imagen por página y una lámina con todos los segmentadores). Son una referencia de diseño dibujada con las cifras del cuaderno; el aspecto final depende de Power BI.

### Segmentadores (filtros)

| Segmentador | Campo | Control | Páginas | Tablas de hechos que filtra |
|---|---|---|---|---|
| Año | `dim_fecha[anio]` | Mosaico, selección múltiple | 1, 2 y 5 (sincronizado) | `fact_solicitud`, `fact_validacion_sintetica` |
| Región | `dim_region[region]` | Desplegable, selección múltiple | 1, 2 y 5 (sincronizado) | `fact_solicitud`, `fact_validacion_sintetica` |
| Tipo de producto | `dim_producto[tipo_producto_financiero]` | Desplegable, selección múltiple | 1, 2 y 5 (sincronizado) | `fact_solicitud`, `fact_validacion_sintetica` |
| Canal | `dim_canal[canal_solicitud]` | Desplegable, selección múltiple | 1 y 2 (la validación no tiene canal) | `fact_solicitud` |
| Nivel de riesgo | `dim_funcionario[nivel_riesgo]` | Mosaico con colores de riesgo | 3 | `fact_funcionario_riesgo` |
| Funcionario | `dim_funcionario[id_funcionario]` | Desplegable con búsqueda; en la página 4, selección única | 3 y 4 | `fact_funcionario_riesgo`, `fact_shap_funcionario` |
| Partición | `fact_validacion_sintetica[particion]` | Mosaico (Entrenamiento / Prueba) | 5 | `fact_validacion_sintetica` |

Configuración: cree año, región y producto en la página 1, cópielos a las páginas 2 y 5 y use **Vista → Sincronizar segmentadores**. En **Formato → Editar interacciones**, desactive la respuesta al segmentador de región en el gráfico de barras por región (página 1) y la respuesta al segmentador de nivel en el gráfico de funcionarios por nivel (página 3). Los segmentadores de región, producto y año **no** cambian los indicadores de funcionarios (K12 a K16), porque esas tablas no tienen esos campos. Para validar los totales con la hoja de Excel, limpie todos los segmentadores.

## 6. Validar los totales

1. Abra `dashboard/Validacion_totales_tablero.xlsx` (hoja «Validación de totales»).
2. En Power BI, sin filtros, cree una tarjeta con cada medida de la columna «Medida» y escriba el valor en la columna H.
3. Para los desgloses (región, producto, canal, estado y año), use una tabla con la dimensión en las filas y `Total de solicitudes`.
4. Las columnas I y J se calculan solas: la diferencia debe ser cero y el resultado «Coincide». Si algún control dice «Revisar», revise primero relaciones, tipos de dato y filtros activos.
5. Tome una captura de la hoja ya diligenciada para el anexo del informe.

## 7. Capturas para el informe

Tome una captura de cada página (1 a 5) y de la hoja de validación diligenciada. Para cada una, el informe necesita título, fuente («Elaboración propia con datos sintéticos»), y una interpretación debajo. No copie cifras a mano: tómelas de `results/tables_obj3/tablero_*.csv`.

## 8. Qué verificó el cuaderno y qué no

- Verificado: las 16 relaciones no tienen filas huérfanas, las claves son únicas, y las medidas calculadas con el equivalente en Python coinciden con la fuente (hoja «Validación de totales», columnas D a G). La explicación SHAP cumple la aditividad con error del orden de 1e-13.
- **No verificado:** la sintaxis DAX y los visuales no se pudieron ejecutar en Power BI desde el entorno de análisis. Por eso la validación final (columnas H a J) la hace el equipo.
- El tablero muestra datos sintéticos; sus resultados no permiten concluir sobre el desempeño en una entidad real.
