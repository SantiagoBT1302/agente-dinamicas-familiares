from databricks import sql
from langchain.tools import tool
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Catálogo de tablas disponibles para el agente
# ─────────────────────────────────────────────────────────────────────────────
# VALORES EXACTOS VERIFICADOS EN DATOS REALES (bronze files auditados):
#
# SEXO:
#   - DANE (dane_personas): sexo = 'Hombre' / 'Mujer'
#   - SISBEN (silver.sisben): sexo_persona = 1 (Hombre) / 2 (Mujer)
#   - SIVIGILA (vigsalpub + intsui): sexo = 'Femenino' / 'Masculino'
#   - ECV (ecv_craccompohog): sexo_nacer = 'Hombre' / 'Mujer'
#
# JEFE DE HOGAR:
#   - DANE 2005: parentesco = 'Jefe(a) del hogar'
#   - DANE 2018: parentesco_jefe_hogar = 'Jefe(a) del hogar'  ← columna DISTINTA
#   - SISBEN: Jefe_UG = 1 (numérico, no texto)
#   - ECV: todos los registros en ecv_craccompohog son jefes (ORDEN=1 siempre)
#
# FILTRO DEPARTAMENTO:
#   - DANE: codigo_departamento = 'Caldas' / 'Quindío' / 'Risaralda'
#   - SIVIGILA vigsalpub: codigo_departamento_ocurrencia = 'Caldas' / 'Quindío' / 'Risaralda'
#   - SIVIGILA intsui: departamento_residencia = 'Caldas' / 'Quindío' / 'Risaralda'  ← DIFERENTE
#   - SISBEN: cod_dpto = 17 (Caldas), 63 (Quindío), 66 (Risaralda)  ← NUMÉRICO
#   - ECV: disponible en datosvivcal via join por DIRECTORIO
#
# TOTALES REALES VERIFICADOS:
#   DANE 2005 Caldas: 889.402 personas, 244.685 hogares
#   DANE 2018 Caldas: 923.472 personas, 309.837 hogares
#   SISBEN IV 2026: Caldas ~253K hogares (55.4% jefas), Quindío ~172K, Risaralda ~290K
#   SIVIGILA vigsalpub: Caldas 2018=2.561 casos / 2024=3.324 casos
#   SIVIGILA intsui: Caldas 2018=1.007 casos / 2024=1.231 casos
#   ECV 2025: Caldas 2.740 hogares (muestra probabilística)
# ─────────────────────────────────────────────────────────────────────────────
TABLA_DESCRIPCION = {
    # ── GOLD (preferir siempre) ──────────────────────────────────────────────
    "gold.jefes_hogar_dane": """Perfil demográfico de jefes/as de hogar DANE Censos 2005 y 2018 — tabla pre-agregada.
Columnas: departamento, año_censo ('2005'/'2018' como STRING), codigo_municipio, sexo ('Hombre'/'Mujer'),
grupo_edad_quinquenal, estado_civil, nivel_educativo, tiene_discapacidad, total_jefes.
⚠️ SIEMPRE usa SUM(total_jefes), NUNCA COUNT(*).
Columnas con ñ o tilde SIEMPRE entre backticks: `año_censo`.
Ejemplo: SELECT sexo, SUM(total_jefes) FROM gold.jefes_hogar_dane
         WHERE departamento='Caldas' AND `año_censo`='2018' GROUP BY sexo
Totales 2018: Caldas ~309K jefes totales. Totales 2005: Caldas ~244K jefes.""",

    "gold.composicion_hogar_dane": """Composición y tamaño de hogares DANE 2005/2018 — tabla pre-agregada por municipio.
Columnas: departamento, año_censo ('2005'/'2018' STRING), codigo_municipio, area_geografica,
total_hogares, promedio_personas_hogar, promedio_cuartos, hogares_unipersonales, hogares_5_o_mas.
⚠️ SIEMPRE usa SUM(total_hogares), NUNCA COUNT(*). Columna `año_censo` requiere backticks.""",

    "gold.jefes_hogar_ecv": """Perfil de jefes/as de hogar ECV 2025 — tabla pre-agregada (muestra ~2.740 hogares en total Eje Cafetero).
Columnas: departamento, sexo_nacer ('Hombre'/'Mujer'), estado_civil, total_jefes, edad_promedio.
⚠️ SIEMPRE usa SUM(total_jefes), NUNCA COUNT(*).
La ECV es muestra probabilística (no censo): representativa pero NO comparable en totales absolutos con DANE o SISBEN.""",

    "gold.fuerza_trabajo_ecv": """Participación laboral jefes de hogar ECV 2025 — tabla pre-agregada.
Columnas: departamento, actividad_semana_pasada, posicion_ocupacional, total_personas.
⚠️ SIEMPRE usa SUM(total_personas), NUNCA COUNT(*).""",

    "gold.sivigila_intsui": """Casos de intento de suicidio SIVIGILA 2018 y 2024 — tabla pre-agregada.
Columnas: departamento, año ('2018'/'2024' STRING en Gold), municipio_residencia, sexo ('Femenino'/'Masculino'),
area_geografica, pertenencia_etnica, tipo_seguridad_social, fue_hospitalizado,
codigo_subgrupo (método: 'Intoxicación por medicamentos'/'Arma cortopunzante'/etc.),
codigo_evento, total_casos, edad_promedio, edad_min, edad_max.
⚠️ SIEMPRE usa SUM(total_casos), NUNCA COUNT(*).
Totales Caldas: 2018=1.007 casos, 2024=1.231 casos. En 2018 y 2024: mujeres son ~64-59% de los casos.""",

    "gold.sivigila_vigsalpub": """Casos de violencia de género e intrafamiliar SIVIGILA 2018 y 2024 — tabla pre-agregada.
Columnas: departamento, año ('2018'/'2024' STRING en Gold), municipio_residencia, sexo ('Femenino'/'Masculino'),
area_geografica, pertenencia_etnica, tipo_seguridad_social, fue_hospitalizado, condicion_final,
codigo_subgrupo (tipo violencia: 'Violencia física'/'Violencia sexual'/'Violencia económica / patrimonial'/
'Acoso sexual'/'Violencia contra niños y adolescentes'/'Abuso sexual'/etc.),
codigo_evento, total_casos, edad_promedio, edad_min, edad_max.
⚠️ SIEMPRE usa SUM(total_casos), NUNCA COUNT(*).
Totales Caldas: 2018=2.561 casos, 2024=3.324 casos (+30%). Víctimas mayoritariamente femeninas (~83% en 2018).""",

    "gold.sisben_municipio": """Conteo de hogares SISBEN IV 2026 por municipio — tabla pre-agregada.
Columnas: departamento, nombre_municipio, clase_territorio ('Cabecera'/'Rural disperso'/'Centro poblado'), total_hogares.
Cubre 26 municipios vulnerables del Eje Cafetero. Filtra por clase_territorio='Cabecera' para comparar municipios.
⚠️ SIEMPRE usa SUM(total_hogares), NUNCA COUNT(*). NO tiene desglose por sexo del jefe — usa silver.sisben para eso.
Totales aprox (todas las clases): Caldas ~253K hogares, Quindío ~172K, Risaralda ~290K.""",

    # ── SILVER (para detalles no disponibles en Gold) ────────────────────────
    "silver.sisben": """SISBEN IV 2026 — datos a nivel de PERSONA (1 fila = 1 persona, ~1.6M filas totales).
⚠️ NO es nivel de hogar: es persona-nivel. Una persona por fila.
Columnas clave:
  - Jefe_UG: 1.0 = es jefe/a del hogar | NULL = no es jefe (filtrar con Jefe_UG = 1)
  - sexo_persona: 1 = Hombre | 2 = Mujer (NUMÉRICO, no texto)
  - cod_dpto: 17 = Caldas | 63 = Quindío | 66 = Risaralda (NUMÉRICO)
  - cod_mpio: código DIVIPOLA numérico (17001=Manizales, 17380=La Dorada, etc.)
  - clase_territorio: 'Cabecera' / 'Rural disperso' / 'Centro poblado' / NULL
  - tip_parentesco: 1=Jefe, 2=Cónyuge, 3=Hijo/a (alternativa a Jefe_UG)
REGLAS PARA CONTAR HOGARES (CRÍTICO):
  ✓ Total hogares: COUNT(*) WHERE Jefe_UG = 1
  ✓ Jefas (Mujer): WHERE Jefe_UG = 1 AND sexo_persona = 2
  ✓ Jefes (Hombre): WHERE Jefe_UG = 1 AND sexo_persona = 1
  ✗ NUNCA: COUNT(*) sin filtro Jefe_UG → cuenta personas, no hogares
Ejemplo correcto para Caldas:
  SELECT COUNT(*) AS total_hogares,
         SUM(CASE WHEN sexo_persona=2 THEN 1 ELSE 0 END) AS jefas,
         SUM(CASE WHEN sexo_persona=1 THEN 1 ELSE 0 END) AS jefes
  FROM silver.sisben WHERE Jefe_UG = 1 AND cod_dpto = 17""",

    "silver.dane_personas": """Personas censadas DANE 2005 y 2018 — nivel de persona (~4.5M registros combinados).
Columnas clave:
  - `año_censo` (STRING, con backticks): '2005' o '2018'
  - sexo: 'Hombre' / 'Mujer' (texto, igual en ambos años)
  - departamento: 'Caldas' / 'Quindío' / 'Risaralda'
  - codigo_municipio: nombre del municipio en texto
  - parentesco (AÑO 2005): 'Jefe(a) del hogar' / 'Hijo(a) / Hijastro(a)' / 'Pareja / Cónyuge / Compañero(a)' / etc.
  - parentesco_jefe_hogar (AÑO 2018): 'Jefe(a) del hogar' / 'Hijo(a), hijastro(a)' / 'Pareja (cónyuge, compañero(a), esposo(a))' / etc.
  ⚠️ CRÍTICO: el nombre de la columna de parentesco CAMBIA entre años:
     - 2005 → columna `parentesco`
     - 2018 → columna `parentesco_jefe_hogar`
  - llave_hogar (2005): ID único de hogar (244.685 únicos para Caldas 2005)
  - numero_hogar_en_vivienda + codigo_encuesta + numero_vivienda + codigo_municipio (2018): clave compuesta de hogar
  - estado_civil: 'Soltero(a)' / 'Casado(a)' / 'Unión libre' / 'Viudo(a)' / 'Separado(a) o divorciado(a)' / etc.
Totales: Caldas 2005=889.402 personas (244.685 hogares) | Caldas 2018=923.472 personas (309.837 hogares)
Ejemplo jefas Caldas 2018:
  SELECT COUNT(*) FROM silver.dane_personas
  WHERE `año_censo`='2018' AND departamento='Caldas'
    AND parentesco_jefe_hogar='Jefe(a) del hogar' AND sexo='Mujer'""",

    "silver.dane_hogares": """Hogares DANE 2005 y 2018 — 1 fila por hogar (~1.4M registros combinados).
Columnas: departamento, `año_censo` (STRING con backticks), codigo_municipio, total_personas_hogar, llave_hogar (2005) / clave compuesta (2018).
JOIN con silver.dane_personas: ON llave_hogar (2005) o ON 4 columnas compuestas (2018).""",

    "silver.dane_viviendas": "Viviendas censadas DANE 2005 y 2018 (~1.5M registros). Condiciones físicas: materiales, servicios públicos, hacinamiento.",

    "silver.ecv_craccompohog": """ECV 2025 — características del hogar y su jefe/a (2.740 hogares muestra, 1 fila = 1 hogar).
⚠️ TODOS los registros son jefes de hogar (ORDEN=1 siempre). No hay datos de otros miembros.
Columnas clave:
  - sexo_nacer: 'Hombre' / 'Mujer' (sexo del jefe/a de hogar)
  - DIRECTORIO: ID único del hogar (2.736 únicos en 2.740 filas)
  - estado_civil, nivel_educativo, edad del jefe
  - parentesco_jefe_hogar: siempre 'Jefe/a del hogar' (no útil para filtrar)
JOIN con otros módulos ECV: ON DIRECTORIO (+ SECUENCIA_P si hay duplicados)
Totales Caldas: 1.571 jefes hombres (57.3%), 1.169 jefas mujeres (42.7%)""",

    "silver.ecv_fuertra": "ECV 2025 — fuerza de trabajo del jefe de hogar (2.740 registros, mismo DIRECTORIO que craccompohog). Columnas: DIRECTORIO, actividad_semana_pasada, posicion_ocupacional, horas_trabajadas, ingresos_mes_pasado.",
    "silver.ecv_salud": "ECV 2025 — salud del jefe de hogar. Columnas: DIRECTORIO, afiliado_salud, regimen_salud.",
    "silver.ecv_educacion": "ECV 2025 — educación. Columnas: DIRECTORIO, sabe_leer_escribir, nivel_educativo.",
    "silver.ecv_condvidhog": "ECV 2025 — condiciones de vida del hogar. Columnas: DIRECTORIO, tipo_vivienda, tenencia_vivienda, acceso_servicios.",
    "silver.ecv_servhog": "ECV 2025 — servicios del hogar (agua, energía, gas, internet). Columnas: DIRECTORIO, tipo_agua, tiene_internet, tiene_gas.",

    "silver.sivigila_intsui": """Intento de suicidio individual SIVIGILA 2018 y 2024 (~5.4K casos totales Eje Cafetero).
Columna `año` (INT): 2018 o 2024 (sin comillas). En Gold `año` es STRING.
Columnas clave:
  - sexo: 'Femenino' / 'Masculino' (NO 'Hombre'/'Mujer')
  - departamento_residencia: 'Caldas' / 'Quindío' / 'Risaralda' ← filtrar por aquí (NO codigo_departamento_ocurrencia)
  - municipio_ocurrencia: nombre del municipio
  - codigo_subgrupo: método — 'Intoxicación por medicamentos' / 'Arma cortopunzante' /
    'Intoxicación por gases y vapores' / 'Intoxicación por plaguicidas' / 'Arma de fuego' / etc.
  - edad, area_geografica, pertenencia_etnica, gp_discapacidad, fue_hospitalizado
  - Solo 2024: estrato_socioeconomico, gp_migrante, nacionalidad
Totales Caldas: 2018=1.007 casos, 2024=1.231 casos""",

    "silver.sivigila_vigsalpub": """Violencia de género e intrafamiliar individual SIVIGILA 2018 y 2024 (~17.2K casos totales).
Columna `año` (INT): 2018 o 2024 (sin comillas). En Gold `año` es STRING.
Columnas clave:
  - sexo: 'Femenino' / 'Masculino' (NO 'Hombre'/'Mujer')
  - codigo_departamento_ocurrencia: 'Caldas' / 'Quindío' / 'Risaralda' ← filtrar por aquí
    (¡DIFERENTE a intsuical que usa departamento_residencia!)
  - municipio_ocurrencia, codigo_municipio_residencia: nombre municipio (texto)
  - codigo_subgrupo: tipo violencia — 'Violencia física' / 'Violencia sexual' /
    'Violencia económica / patrimonial' / 'Acoso sexual' / 'Abuso sexual' /
    'Violencia contra niños y adolescentes' / 'Violencia física extrafamiliar' / etc.
  - condicion_final: estado del caso
  - edad, area_geografica, pertenencia_etnica, fue_hospitalizado, tipo_seguridad_social
  - Solo 2024: estrato_socioeconomico, gp_migrante, gp_desmovilizado, nacionalidad
Totales Caldas: 2018=2.561 casos, 2024=3.324 casos. ~83% víctimas femeninas.""",

    # ── BRONZE ───────────────────────────────────────────────────────────────
    "bronze.sisben_caldas": "SISBEN IV Caldas — persona-nivel (605.843 personas, ~253K hogares, 124 cols). Mismas columnas que silver.sisben: Jefe_UG (1=jefe), sexo_persona (1=H, 2=M), cod_dpto=17.",
    "bronze.sisben_quindio": "SISBEN IV Quindío — persona-nivel (~172K hogares). Mismas columnas que silver.sisben.",
    "bronze.sisben_risaralda": "SISBEN IV Risaralda — persona-nivel (~290K hogares). Mismas columnas que silver.sisben.",

    # ── DICCIONARIOS ─────────────────────────────────────────────────────────
    "gold.diccionarios": "Diccionario de datos con nombres REALES de columnas de tablas Silver y Gold. Columnas: fuente, tabla, columna, tipo_dato, tipo_columna, valores_posibles, n_valores_unicos.",
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
