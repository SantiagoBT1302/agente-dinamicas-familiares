from databricks import sql
from langchain.tools import tool
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Catálogo de tablas disponibles para el agente
TABLA_DESCRIPCION = {
    # GOLD — preferir siempre para eficiencia
    "gold.jefes_hogar_dane": "Perfil demográfico de jefes/as de hogar por departamento y municipio (DANE Censos 2005 y 2018). Columnas: departamento, año_censo, codigo_municipio, sexo, grupo_edad_quinquenal, estado_civil, nivel_educativo, tiene_discapacidad, total_jefes. IMPORTANTE: es una tabla pre-agregada — para obtener totales usa SIEMPRE SUM(total_jefes), NUNCA COUNT(*). Ejemplo: SELECT SUM(total_jefes) FROM gold.jefes_hogar_dane WHERE departamento='Risaralda' AND sexo='Mujer' AND `año_censo`='2018'",
    "gold.composicion_hogar_dane": "Composición y tamaño de hogares por municipio (DANE). Columnas: departamento, año_censo, codigo_municipio, area_geografica, total_hogares, promedio_personas_hogar, promedio_cuartos, hogares_unipersonales, hogares_5_o_mas.",
    "gold.jefes_hogar_ecv": "Perfil de jefes/as de hogar según ECV 2025. Columnas: departamento, sexo_nacer, estado_civil, total_jefes, edad_promedio.",
    "gold.fuerza_trabajo_ecv": "Participación laboral por departamento (ECV 2025). Columnas: departamento, actividad_semana_pasada, posicion_ocupacional, total_personas.",
    "gold.sivigila_intsui": "Casos de intento de suicidio por municipio, sexo y método (SIVIGILA 2018 y 2024). Columnas: departamento, año, municipio_residencia, sexo, area_geografica, pertenencia_etnica, tipo_seguridad_social, fue_hospitalizado, codigo_subgrupo, codigo_evento, total_casos, edad_promedio, edad_min, edad_max. Filtrar por año='2018' o año='2024'.",
    "gold.sivigila_vigsalpub": "Casos de violencia (VCM, VIF, VSX, violencia intrafamiliar) por municipio (SIVIGILA 2018 y 2024). Columnas: departamento, año, municipio_residencia, sexo, area_geografica, pertenencia_etnica, tipo_seguridad_social, fue_hospitalizado, condicion_final, codigo_subgrupo, codigo_evento, total_casos, edad_promedio, edad_min, edad_max. Filtrar por año='2018' o año='2024'.",
    "gold.sisben_municipio": "Conteo de hogares SISBEN IV por municipio (2026). Columnas: departamento, nombre_municipio, clase_territorio (Cabecera/Rural disperso/Centro poblado), total_hogares. Cubre 26 municipios vulnerables del Eje Cafetero. Filtra por clase_territorio='Cabecera' para comparaciones entre municipios.",
    # SILVER — para consultas más detalladas
    "silver.sisben": "Datos SISBEN IV individuales de los 3 departamentos (1.6M hogares). Para consultas detalladas por municipio o variables específicas.",
    "silver.dane_personas": "Personas censadas DANE 2005 y 2018 (4.5M registros). Para análisis demográficos detallados.",
    "silver.dane_hogares": "Hogares censados DANE 2005 y 2018 (1.4M registros).",
    "silver.dane_viviendas": "Viviendas censadas DANE 2005 y 2018 (1.5M registros). Condiciones físicas de la vivienda.",
    "silver.ecv_craccompohog": "Características y composición del hogar ECV 2025. Contiene estado civil, parentesco, edad.",
    "silver.ecv_fuertra": "Fuerza de trabajo ECV 2025. Actividad laboral y ocupación.",
    "silver.ecv_salud": "Módulo de salud ECV 2025.",
    "silver.ecv_educacion": "Módulo de educación ECV 2025.",
    "silver.ecv_condvidhog": "Condiciones de vida del hogar ECV 2025.",
    "silver.ecv_servhog": "Servicios del hogar ECV 2025 (agua, energía, gas, internet).",
    "silver.sivigila_intsui": "Intento de suicidio individual SIVIGILA 2018 y 2024 (~5.4K casos). Columna año (int): 2018 o 2024. Incluye: sexo, edad, municipio_residencia, departamento, area_geografica, pertenencia_etnica, gp_discapacidad, gp_gestante, tipo_seguridad_social, codigo_subgrupo, fue_hospitalizado. Cols exclusivas de 2024: estrato_socioeconomico, gp_migrante, nacionalidad.",
    "silver.sivigila_vigsalpub": "Violencia de género e intrafamiliar individual SIVIGILA 2018 y 2024 (~17.2K casos). Columna año (int): 2018 o 2024. Incluye: sexo, edad, municipio_residencia, departamento, area_geografica, pertenencia_etnica, tipo_seguridad_social, codigo_subgrupo, condicion_final, fue_hospitalizado. Cols exclusivas de 2024: estrato_socioeconomico, gp_migrante, gp_desmovilizado, nacionalidad.",
    # BRONZE — datos crudos
    "bronze.sisben_caldas": "Datos crudos SISBEN IV Caldas (255K hogares, 124 columnas). Fuente primaria.",
    "bronze.sisben_quindio": "Datos crudos SISBEN IV Quindío (171K hogares).",
    "bronze.sisben_risaralda": "Datos crudos SISBEN IV Risaralda (188K hogares).",
    # GOLD — diccionarios
    "gold.diccionarios": "Diccionario de datos con los nombres REALES de columnas de todas las tablas Silver y Gold. Columnas: fuente, tabla, columna, tipo_dato, tipo_columna, valores_posibles, n_valores_unicos. Usar con search_dictionary.",
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
