from databricks import sql
from langchain.tools import tool
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Catálogo de tablas — solo estructura y nombres de columnas.
# Los totales y porcentajes se obtienen siempre consultando Databricks en tiempo real.
# ─────────────────────────────────────────────────────────────────────────────
# GUÍA DE COLUMNAS CLAVE (para escribir SQL correcto):
#
# SEXO — el valor y nombre de columna varía por fuente:
#   DANE:                 sexo = 'Hombre' / 'Mujer'
#   SISBEN Caldas:        sexo_persona = 1 (Hombre) / 2 (Mujer)  ← NUMÉRICO
#   SISBEN Quindío/Ris.:  sexo = 'Hombre' / 'Mujer'              ← TEXTO
#   SIVIGILA:             sexo = 'Femenino' / 'Masculino'
#   ECV:                  sexo_nacer = 'Hombre' / 'Mujer'
#
# JEFE DE HOGAR — varía por fuente:
#   DANE 2005:            parentesco = 'Jefe(a) del hogar'
#   DANE 2018:            parentesco_jefe_hogar = 'Jefe(a) del hogar'  ← columna DISTINTA
#   SISBEN Caldas:        Jefe_UG = 1 (numérico)
#   SISBEN Quindío/Ris.:  parentesco_jefe_hogar = 'Jefe del hogar'     ← TEXTO
#   ECV craccompohog:     todos los registros son jefes (ORDEN=1 siempre)
#
# FILTRO DEPARTAMENTO — varía por fuente:
#   DANE:                 departamento = 'Caldas' / 'Quindío' / 'Risaralda'
#   SISBEN silver:        cod_dpto = 17 / 63 / 66  ← NUMÉRICO
#   SIVIGILA vigsalpub:   codigo_departamento_ocurrencia = 'Caldas' / 'Quindío' / 'Risaralda'
#   SIVIGILA intsui:      departamento_residencia = 'Caldas' / 'Quindío' / 'Risaralda' ← DISTINTO
#   ECV:                  departamento = 'Caldas' / 'Quindío' / 'Risaralda'
# ─────────────────────────────────────────────────────────────────────────────
TABLA_DESCRIPCION = {
    # ── GOLD — tablas pre-agregadas (preferir siempre) ───────────────────────
    "gold.jefes_hogar_dane": """Jefes/as de hogar DANE 2005 y 2018 — tabla pre-agregada, 3 departamentos.
Columnas: departamento, año_censo (STRING: '2005'/'2018'), codigo_municipio, sexo ('Hombre'/'Mujer'),
grupo_edad_quinquenal, estado_civil, nivel_educativo, tiene_discapacidad, total_jefes.
⚠️ SIEMPRE SUM(total_jefes), NUNCA COUNT(*). Columnas con ñ/tilde entre backticks: `año_censo`.""",

    "gold.composicion_hogar_dane": """Composición de hogares DANE 2005/2018 — tabla pre-agregada, 3 departamentos.
Columnas: departamento, año_censo (STRING: '2005'/'2018'), codigo_municipio, area_geografica,
total_hogares, promedio_personas_hogar, promedio_cuartos, hogares_unipersonales, hogares_5_o_mas.
⚠️ SIEMPRE SUM(total_hogares), NUNCA COUNT(*). Columna `año_censo` requiere backticks.""",

    "gold.jefes_hogar_ecv": """Jefes/as de hogar ECV 2025 — tabla pre-agregada, 3 departamentos (muestra probabilística).
Columnas: departamento, sexo_nacer ('Hombre'/'Mujer'), estado_civil, total_jefes, edad_promedio.
⚠️ SIEMPRE SUM(total_jefes), NUNCA COUNT(*).
La ECV es muestra — sus totales absolutos NO son comparables con DANE ni SISBEN.""",

    "gold.fuerza_trabajo_ecv": """Participación laboral ECV 2025 — tabla pre-agregada, 3 departamentos.
Columnas: departamento, actividad_semana_pasada, posicion_ocupacional, total_personas.
⚠️ SIEMPRE SUM(total_personas), NUNCA COUNT(*).""",

    "gold.sivigila_intsui": """Intentos de suicidio SIVIGILA 2018 y 2024 — tabla pre-agregada, 3 departamentos.
Columnas: departamento, año (STRING: '2018'/'2024'), municipio_residencia, sexo ('Femenino'/'Masculino'),
area_geografica, pertenencia_etnica, tipo_seguridad_social, fue_hospitalizado,
codigo_subgrupo (método: 'Intoxicación por medicamentos', 'Arma cortopunzante', etc.),
codigo_evento, total_casos, edad_promedio, edad_min, edad_max.
⚠️ SIEMPRE SUM(total_casos), NUNCA COUNT(*).""",

    "gold.sivigila_vigsalpub": """Violencia de género e intrafamiliar SIVIGILA 2018 y 2024 — tabla pre-agregada, 3 departamentos.
Columnas: departamento, año (STRING: '2018'/'2024'), municipio_residencia, sexo ('Femenino'/'Masculino'),
area_geografica, pertenencia_etnica, tipo_seguridad_social, fue_hospitalizado, condicion_final,
codigo_subgrupo (tipo: 'Violencia física'/'Violencia sexual'/'Violencia económica / patrimonial'/
'Acoso sexual'/'Abuso sexual'/'Violencia contra niños y adolescentes'/etc.),
codigo_evento, total_casos, edad_promedio, edad_min, edad_max.
⚠️ SIEMPRE SUM(total_casos), NUNCA COUNT(*).""",

    "gold.sisben_municipio": """Hogares SISBEN IV 2026 por municipio — tabla pre-agregada, 3 departamentos.
Columnas: departamento, nombre_municipio, clase_territorio ('Cabecera'/'Rural disperso'/'Centro poblado'), total_hogares.
⚠️ SIEMPRE SUM(total_hogares), NUNCA COUNT(*). No tiene desglose por sexo del jefe.""",

    # ── SILVER — datos individuales ──────────────────────────────────────────
    "silver.sisben": """SISBEN IV 2026 — nivel de PERSONA (1 fila = 1 persona). Contiene datos de Caldas (cod_dpto=17).
⚠️ Para obtener hogares SIEMPRE filtra WHERE Jefe_UG = 1 (cada hogar tiene exactamente 1 jefe).
Columnas clave:
  - Jefe_UG: 1.0 = jefe/a del hogar | NULL = no es jefe
  - sexo_persona: 1 = Hombre | 2 = Mujer (NUMÉRICO)
  - cod_dpto: 17 = Caldas | 63 = Quindío | 66 = Risaralda (NUMÉRICO)
  - cod_mpio: código DIVIPOLA numérico
  - clase_territorio: 'Cabecera' / 'Rural disperso' / 'Centro poblado'
  - tip_parentesco: 1=Jefe, 2=Cónyuge, 3=Hijo/a
⛔ NO tiene columnas de clasificación de pobreza (Grupo / clasificacion_sisben_iv).
   Para análisis de pobreza SIEMPRE usa las tablas BRONZE (bronze.sisben_caldas, bronze.sisben_quindio, bronze.sisben_risaralda).
   Consultar silver.sisben para pobreza retornará errores o ceros incorrectos.
Ejemplo para jefaturas femeninas en Caldas:
  SELECT COUNT(*) AS total_hogares,
         SUM(CASE WHEN sexo_persona=2 THEN 1 ELSE 0 END) AS jefas
  FROM silver.sisben WHERE Jefe_UG = 1 AND cod_dpto = 17""",

    "silver.dane_personas": """Personas censadas DANE — 1 fila = 1 persona, cubre 2005 y 2018, 3 departamentos.
Columnas clave:
  - `año_censo` (STRING, siempre entre backticks): '2005' o '2018'
  - sexo: 'Hombre' / 'Mujer'
  - departamento: 'Caldas' / 'Quindío' / 'Risaralda'
  - codigo_municipio: nombre del municipio
  ⚠️ La columna de parentesco cambia entre años:
     2005 → parentesco = 'Jefe(a) del hogar'
     2018 → parentesco_jefe_hogar = 'Jefe(a) del hogar'
  - estado_civil: 'Soltero(a)' / 'Casado(a)' / 'Unión libre' / 'Viudo(a)' / 'Separado(a) o divorciado(a)'
  - llave_hogar (solo 2005): ID único de hogar
  - clave compuesta 2018: numero_hogar_en_vivienda + codigo_encuesta + numero_vivienda + codigo_municipio""",

    "silver.dane_hogares": """Hogares DANE 2005 y 2018 — 1 fila = 1 hogar, 3 departamentos.
Columnas: departamento, `año_censo` (STRING, backticks), codigo_municipio, total_personas_hogar.
JOIN con silver.dane_personas: ON llave_hogar (2005) o clave compuesta de 4 columnas (2018).""",

    "silver.dane_viviendas": "Viviendas DANE 2005 y 2018 — condiciones físicas, materiales, servicios públicos, hacinamiento. 3 departamentos.",

    "silver.ecv_craccompohog": """ECV 2025 — 1 fila = 1 hogar (exclusivamente jefes de hogar, ORDEN=1 siempre). 3 departamentos.
Columnas clave:
  - sexo_nacer: 'Hombre' / 'Mujer'
  - DIRECTORIO: ID único del hogar (usar para JOIN con otros módulos ECV)
  - departamento: 'Caldas' / 'Quindío' / 'Risaralda'
  - estado_civil, nivel_educativo, edad, satisfaccion_ingreso
JOIN con otros módulos ECV: ON DIRECTORIO""",

    "silver.ecv_condvidhog": """ECV 2025 — condiciones de vida y percepción de pobreza. 1 fila = 1 hogar. 3 departamentos.
JOIN con craccompohog ON DIRECTORIO.
Columnas clave de pobreza y bienestar:
  - se_considera_pobre: 'Sí' / 'No'  ← pobreza subjetiva
    ⚠️ Verificar encoding real: SELECT DISTINCT se_considera_pobre FROM silver.ecv_condvidhog LIMIT 5
  - situacion_ingresos_hogar: 'Alcanzan para cubrir gastos mínimos' /
    'No alcanzan para cubrir gastos mínimos' / 'Cubren más que los gastos mínimos'
    ⚠️ Verificar valores reales con SELECT DISTINCT antes de filtrar
  - recibe_subsidio_gobierno / subsidio_colombia_mayor / subsidio_renta_ciudadana_hambre
  - percepcion_economia_hogar_vs_hace_12m
  - eventos_adversos_hogar / evento_jefe_perdio_empleo / evento_cierre_negocio
  - alim_salto_comida / alim_comio_menos / alim_hogar_sin_alimentos / alim_tuvo_hambre_sin_comer
  - departamento: 'Caldas' / 'Quindío' / 'Risaralda'""",

    "silver.ecv_fuertra": "ECV 2025 — fuerza de trabajo. 3 departamentos. Columnas: DIRECTORIO, departamento, actividad_semana_pasada, posicion_ocupacional, horas_trabajadas, ingresos_mes_pasado.",
    "silver.ecv_salud": "ECV 2025 — salud. 3 departamentos. Columnas: DIRECTORIO, departamento, afiliado_salud, regimen_salud.",
    "silver.ecv_educacion": "ECV 2025 — educación. 3 departamentos. Columnas: DIRECTORIO, departamento, sabe_leer_escribir, nivel_educativo.",
    "silver.ecv_servhog": "ECV 2025 — servicios del hogar. 3 departamentos. Columnas: DIRECTORIO, departamento, tipo_agua, tiene_internet, tiene_gas.",

    "silver.sivigila_intsui": """Intentos de suicidio individuales SIVIGILA 2018 y 2024. 3 departamentos.
Columna `año` (INT, sin comillas): 2018 o 2024.
Columnas clave:
  - sexo: 'Femenino' / 'Masculino'
  - departamento_residencia: 'Caldas' / 'Quindío' / 'Risaralda'  ← filtrar aquí
  - municipio_ocurrencia, codigo_subgrupo (método), edad, area_geografica
  - fue_hospitalizado, pertenencia_etnica, gp_discapacidad
  - Solo 2024: estrato_socioeconomico, gp_migrante, nacionalidad""",

    "silver.sivigila_vigsalpub": """Violencia de género e intrafamiliar individual SIVIGILA 2018 y 2024. 3 departamentos.
Columna `año` (INT, sin comillas): 2018 o 2024.
Columnas clave:
  - sexo: 'Femenino' / 'Masculino'
  - codigo_departamento_ocurrencia: 'Caldas' / 'Quindío' / 'Risaralda'  ← filtrar aquí
    ⚠️ DIFERENTE a silver.sivigila_intsui que usa departamento_residencia
  - municipio_ocurrencia, codigo_subgrupo (tipo de violencia), condicion_final, edad
  - fue_hospitalizado, area_geografica, pertenencia_etnica, tipo_seguridad_social
  - Solo 2024: estrato_socioeconomico, gp_migrante, gp_desmovilizado, nacionalidad""",

    # ── BRONZE — datos crudos por departamento ───────────────────────────────
    "bronze.sisben_caldas": """SISBEN IV Caldas — nivel de PERSONA, 124 columnas.
Columnas jefe/sexo/clasificación:
  - Jefe_UG: 1.0 = jefe | NULL = no jefe
  - sexo_persona: 1 = Hombre | 2 = Mujer (NUMÉRICO)
  - Grupo: 'A'=Pobreza extrema / 'B'=Pobreza moderada / 'C'=Vulnerable / 'D'=No pobre
  - Clasificacion: nivel dentro del grupo ('A1'-'A5', 'B1'-'B7', 'C1'-'C18', 'D1'-'D21')
Para hogares en pobreza: WHERE Jefe_UG = 1 AND Grupo = 'A' (extrema) o Grupo IN ('A','B') (pobres)""",

    "bronze.sisben_quindio": """SISBEN IV Quindío — nivel de PERSONA, 91 columnas.
⚠️ Estructura DIFERENTE a bronze.sisben_caldas:
  - Columna jefe: parentesco_jefe_hogar = 'Jefe del hogar' (TEXTO)
  - Columna sexo: sexo = 'Hombre' / 'Mujer' (TEXTO)
  - Columna clasificación: clasificacion_sisben_iv ('A1'-'D21')
    → Extraer grupo: SUBSTRING(clasificacion_sisben_iv, 1, 1) IN ('A','B','C','D')
Para hogares en pobreza: WHERE parentesco_jefe_hogar='Jefe del hogar' AND SUBSTRING(clasificacion_sisben_iv,1,1)='A'""",

    "bronze.sisben_risaralda": """SISBEN IV Risaralda — nivel de PERSONA, 91 columnas.
Misma estructura que bronze.sisben_quindio (diferente a bronze.sisben_caldas):
  - Columna jefe: parentesco_jefe_hogar = 'Jefe del hogar' (TEXTO)
  - Columna sexo: sexo = 'Hombre' / 'Mujer' (TEXTO)
  - Columna clasificación: clasificacion_sisben_iv ('A1'-'D21')
    → Extraer grupo: SUBSTRING(clasificacion_sisben_iv, 1, 1)""",

    # ── DICCIONARIOS ─────────────────────────────────────────────────────────
    "gold.diccionarios": "Diccionario de columnas de tablas Silver y Gold. Columnas: fuente, tabla, columna, tipo_dato, tipo_columna, valores_posibles, n_valores_unicos.",
}

def _get_connection():
    return sql.connect(
        server_hostname=settings.databricks_host.replace("https://", ""),
        http_path=settings.databricks_http_path,
        access_token=settings.databricks_token,
    )

@tool
def query_databricks(sql_query: str) -> str:
    """
    Ejecuta una consulta SQL en el lakehouse de Databricks y devuelve los resultados.
    Usa tablas Gold por defecto (gold.*) para eficiencia.
    Para consultas detalladas usa Silver (silver.*).
    Siempre incluye LIMIT en las consultas para evitar resultados masivos.
    """
    try:
        with _get_connection() as conn:
            with conn.cursor() as cursor:
                logger.info(f"Ejecutando SQL: {sql_query}")
                cursor.execute(sql_query)
                results = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

                if not results:
                    return "La consulta no devolvió resultados."

                # Formatear como tabla legible
                rows = [dict(zip(columns, row)) for row in results]
                output = f"Resultados ({len(rows)} filas):\n"
                output += " | ".join(columns) + "\n"
                output += "-" * 80 + "\n"
                for row in rows[:50]:  # máximo 50 filas al agente
                    output += " | ".join(str(v) for v in row.values()) + "\n"
                if len(rows) > 50:
                    output += f"... ({len(rows) - 50} filas adicionales omitidas)"
                return output

    except Exception as e:
        logger.error(f"Error ejecutando SQL: {e}")
        return f"Error en la consulta: {str(e)}"

@tool
def list_tables() -> str:
    """
    Lista todas las tablas disponibles en el lakehouse con su descripción.
    Úsalo cuando necesites saber qué datos están disponibles.
    """
    output = "Tablas disponibles en el lakehouse:\n\n"
    for tabla, desc in TABLA_DESCRIPCION.items():
        capa = tabla.split(".")[0].upper()
        output += f"[{capa}] {tabla}\n  → {desc}\n\n"
    return output

@tool
def get_schema(table_name: str) -> str:
    """
    Devuelve el esquema (columnas y tipos) de una tabla específica.
    Úsalo antes de escribir una consulta SQL para conocer las columnas exactas.
    """
    try:
        with _get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"DESCRIBE {table_name}")
                results = cursor.fetchall()
                output = f"Esquema de {table_name}:\n"
                for row in results:
                    output += f"  {row[0]:40s} {row[1]}\n"
                return output
    except Exception as e:
        return f"Error obteniendo esquema de {table_name}: {str(e)}"

# ─────────────────────────────────────────────────────────────────────────────
# Mapa de columnas Silver/Gold → términos de búsqueda en los diccionarios.
#
# IMPORTANTE: Los diccionarios usan los NOMBRES ORIGINALES de las encuestas
# (códigos ECV: P6020, P6040, P6051…; nombres DANE originales; etc.).
# Las tablas Silver/Gold tienen columnas RENOMBRADAS (sexo_nacer, edad_anos…).
# Las columnas Gold derivadas (grupo_edad_quinquenal, total_jefes…) no existen
# en los diccionarios porque son calculadas en el proceso ETL.
# Este mapa traduce nombres Silver/Gold → término buscable en el diccionario.
# ─────────────────────────────────────────────────────────────────────────────
COLUMN_SEARCH_MAP = {
    # ECV — craccompohog
    "sexo_nacer":                 ["P6020", "sexo al nacer"],
    "edad_anos":                  ["P6040", "años cumplidos", "edad"],
    "parentesco_jefe_hogar":      ["P6051", "parentesco"],
    "estado_civil":               ["P5502", "estado civil"],
    "conyuge_vive_hogar":         ["P6071", "cónyuge"],
    "municipio_nacimiento":       ["P756", "nacimiento", "municipio"],
    "siempre_vivio_municipio":    ["P6074", "siempre vivió"],
    "residencia_hace_5anos":      ["P755", "hace 5 años"],
    "pertenencia_etnica":         ["P6080", "étnica", "étnico"],
    "nivel_educacion_padre":      ["P6087", "educación padre"],
    "nivel_educacion_madre":      ["P6088", "educación madre"],
    # ECV — fuerza de trabajo
    "actividad_semana_pasada":    ["actividad semana pasada", "P6240"],
    "posicion_ocupacional":       ["posición ocupacional", "P6430"],
    "tipo_contrato":              ["tipo de contrato", "P6460"],
    "horas_trabajadas":           ["horas trabajadas", "P6800"],
    "ingresos_mes_pasado":        ["ingresos", "P6500"],
    # ECV — salud
    "afiliado_salud":             ["afiliado", "sistema de salud", "P6090"],
    "regimen_salud":              ["régimen", "P6100"],
    # ECV — educacion
    "sabe_leer_escribir":         ["alfabetismo", "leer y escribir", "P6160"],
    "asiste_establecimiento":     ["asistencia escolar", "P6170"],
    "nivel_educativo":            ["nivel educativo", "P6210"],
    # ECV columnas Gold derivadas (no están en diccionarios — conceptos base)
    "grupo_edad_quinquenal":      ["edad", "P6040", "años cumplidos"],
    "edad_promedio":              ["edad", "P6040"],
    "total_jefes":                ["parentesco", "P6051", "jefe"],
    "total_personas":             ["actividad", "P6240"],
    # DANE censos
    "sexo":                       ["sexo", "P_SEXO"],
    "tiene_discapacidad":         ["discapacidad", "P_DISC"],
    "area_geografica":            ["área", "P_ZONA"],
    "total_hogares":              ["hogares"],
    "codigo_municipio":           ["municipio", "DIVIPOLA"],
    # SISBEN
    "cod_mpio":                   ["municipio", "código municipio"],
    "clase_territorio":           ["clase", "territorio", "zona"],
    "gasto_prom_alimento":        ["gasto", "alimento"],
    "gasto_prom_salud":           ["gasto", "salud"],
    "gasto_prom_educacion":       ["gasto", "educación"],
    "gasto_prom_servicios":       ["gasto", "servicios"],
    # SIVIGILA
    "municipio_residencia":       ["municipio residencia"],
    "total_casos":                ["casos", "notificación"],
    "codigo_subgrupo":            ["subgrupo", "tipo violencia", "método", "intoxicación"],
    "condicion_final":            ["condición", "fallecido", "vivo", "muerte"],
    "fue_hospitalizado":          ["hospitalizado", "hospitalización"],
    "gp_discapacidad":            ["discapacidad", "grupo discapacidad"],
    "gp_gestante":                ["gestante", "embarazo"],
    "gp_victima_violencia":       ["víctima violencia"],
    "gp_desplazado":              ["desplazado"],
    "gp_migrante":                ["migrante"],
    "gp_trastorno_psiquiatrico":  ["trastorno psiquiátrico", "psiquiátrico"],
    "estrato_socioeconomico":     ["estrato", "socioeconómico"],
    "pertenencia_etnica":         ["étnica", "étnico", "indígena", "afrocolombiano"],
}

@tool
def search_dictionary(query: str) -> str:
    """
    Busca en gold.diccionarios el significado de una columna: su tipo de dato,
    valores posibles (columnas categóricas) o rango (columnas numéricas).

    Los diccionarios están construidos con los NOMBRES REALES de las columnas
    de las tablas Silver y Gold, por lo que puedes buscar directamente por
    nombre de columna (ej: 'sexo_nacer', 'estado_civil', 'parentesco_jefe_hogar').

    También puedes buscar por concepto (ej: 'sexo', 'edad', 'parentesco')
    para ver todas las columnas relacionadas en todas las tablas.

    Ejemplos:
    - 'sexo_nacer'            → valores posibles en ecv_craccompohog
    - 'parentesco_jefe_hogar' → categorías de parentesco
    - 'actividad_semana'      → opciones de actividad laboral
    - 'estado_civil'          → categorías de estado civil
    """
    query_safe = query.replace("'", "''")
    try:
        with _get_connection() as conn:
            with conn.cursor() as cursor:
                # Búsqueda 1: coincidencia exacta por nombre de columna
                sql_exacto = f"""
                    SELECT fuente, tabla, columna, tipo_dato, tipo_columna, valores_posibles, n_valores_unicos
                    FROM gold.diccionarios
                    WHERE LOWER(columna) = LOWER('{query_safe}')
                    ORDER BY tabla
                    LIMIT 10
                """
                cursor.execute(sql_exacto)
                resultados = cursor.fetchall()

                # Búsqueda 2: si no hay exacta, búsqueda parcial en nombre de columna
                if not resultados:
                    sql_parcial = f"""
                        SELECT fuente, tabla, columna, tipo_dato, tipo_columna, valores_posibles, n_valores_unicos
                        FROM gold.diccionarios
                        WHERE LOWER(columna) LIKE LOWER('%{query_safe}%')
                           OR LOWER(valores_posibles) LIKE LOWER('%{query_safe}%')
                        ORDER BY
                            CASE WHEN LOWER(columna) LIKE LOWER('%{query_safe}%') THEN 0 ELSE 1 END,
                            tabla, columna
                        LIMIT 15
                    """
                    cursor.execute(sql_parcial)
                    resultados = cursor.fetchall()

                if not resultados:
                    return (
                        f"No se encontró '{query}' en gold.diccionarios.\n"
                        f"Intenta con un término más general (ej: 'sexo', 'edad', 'salud')."
                    )

                cols = ["fuente", "tabla", "columna", "tipo_dato", "tipo_columna", "valores_posibles", "n_valores_unicos"]
                output = f"Diccionario para '{query}' ({len(resultados)} resultado(s)):\n\n"
                for row in resultados:
                    r = dict(zip(cols, row))
                    output += (
                        f"[{r['fuente']}] {r['tabla']} → {r['columna']}\n"
                        f"  Tipo: {r['tipo_dato']} | Categoría: {r['tipo_columna']} | N únicos: {r['n_valores_unicos']}\n"
                        f"  Valores: {r['valores_posibles']}\n"
                        f"{'─'*60}\n"
                    )
                return output

    except Exception as e:
        return f"Error buscando en diccionarios: {str(e)}"

TOOLS = [query_databricks, list_tables, get_schema, search_dictionary]
