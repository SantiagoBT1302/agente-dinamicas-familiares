from databricks import sql
from langchain.tools import tool
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Catálogo de tablas disponibles para el agente
# ─────────────────────────────────────────────────────────────────────────────
# VALORES EXACTOS VERIFICADOS EN ARCHIVOS FUENTE (auditoría completa):
#
# SEXO:
#   - DANE (dane_personas): sexo = 'Hombre' / 'Mujer'
#   - SISBEN Caldas (bronze.sisben_caldas): sexo_persona = 1 (Hombre) / 2 (Mujer)  ← NUMÉRICO
#   - SISBEN Quindío/Risaralda (bronze.sisben_quindio/risaralda): sexo = 'Hombre' / 'Mujer'  ← TEXTO
#   - SIVIGILA (vigsalpub + intsui): sexo = 'Femenino' / 'Masculino'
#   - ECV (ecv_craccompohog): sexo_nacer = 'Hombre' / 'Mujer'
#
# JEFE DE HOGAR:
#   - DANE 2005: parentesco = 'Jefe(a) del hogar'
#   - DANE 2018: parentesco_jefe_hogar = 'Jefe(a) del hogar'  ← columna DISTINTA
#   - SISBEN Caldas: Jefe_UG = 1 (numérico)
#   - SISBEN Quindío/Risaralda: parentesco_jefe_hogar = 'Jefe del hogar'  ← TEXTO, columna DISTINTA
#   - ECV: todos los registros en ecv_craccompohog son jefes (ORDEN=1 siempre)
#
# FILTRO DEPARTAMENTO:
#   - DANE: departamento = 'Caldas' / 'Quindío' / 'Risaralda'
#   - SIVIGILA vigsalpub: codigo_departamento_ocurrencia = 'Caldas' / 'Quindío' / 'Risaralda'
#   - SIVIGILA intsui: departamento_residencia = 'Caldas' / 'Quindío' / 'Risaralda'  ← DIFERENTE
#   - SISBEN silver: cod_dpto = 17 (Caldas), 63 (Quindío), 66 (Risaralda)  ← NUMÉRICO
#   - ECV: ver columna departamento en cada tabla (una por departamento)
#
# TOTALES EXACTOS VERIFICADOS:
#   SISBEN Caldas:    605.843 personas | 253.243 hogares (Jefe_UG=1) | 55.4% jefas
#   SISBEN Quindío:   366.319 personas | 171.903 hogares              | 59.4% jefas
#   SISBEN Risaralda: 641.615 personas | 289.815 hogares              | 61.2% jefas
#   DANE Caldas 2005:    889.402 personas | 244.685 hogares | 30.5% jefas
#   DANE Caldas 2018:    923.472 personas | 309.837 hogares | 38.8% jefas
#   DANE Quindío 2005:   514.747 personas | 142.982 hogares | 32.7% jefas
#   DANE Quindío 2018:   509.640 personas | 174.475 hogares | 41.7% jefas
#   DANE Risaralda 2005: 853.697 personas | 230.532 hogares | 31.5% jefas
#   DANE Risaralda 2018: 839.597 personas | 278.133 hogares | 42.6% jefas
#   SIVIGILA vigsalpub: Caldas 2018=2.561 / 2024=3.324 | Quindío 2018=1.950 / 2024=2.687 | Risaralda 2018=2.975 / 2024=3.718
#   SIVIGILA intsui:    Caldas 2018=1.007 / 2024=1.231 | Quindío 2018=570 / 2024=493 | Risaralda 2018=793 / 2024=1.260
#   ECV craccompohog:   Caldas=2.740 hogares (42.7% jefas) | Quindío=2.791 (42.5%) | Risaralda=2.828 (42.7%)
# ─────────────────────────────────────────────────────────────────────────────
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
    "gold.jefes_hogar_dane": """Perfil demográfico de jefes/as de hogar DANE Censos 2005 y 2018 — tabla pre-agregada (3 departamentos).
Columnas: departamento, año_censo ('2005'/'2018' como STRING), codigo_municipio, sexo ('Hombre'/'Mujer'),
grupo_edad_quinquenal, estado_civil, nivel_educativo, tiene_discapacidad, total_jefes.
⚠️ SIEMPRE usa SUM(total_jefes), NUNCA COUNT(*). Columnas con ñ/tilde entre backticks: `año_censo`.
TOTALES EXACTOS (jefes de hogar):
  Caldas    2005: 244.685 jefes (30.5% mujeres) | 2018: 309.680 jefes (38.8% mujeres)
  Quindío   2005: 142.982 jefes (32.7% mujeres) | 2018: 174.231 jefes (41.7% mujeres)
  Risaralda 2005: 230.532 jefes (31.5% mujeres) | 2018: 277.932 jefes (42.6% mujeres)""",

    "gold.composicion_hogar_dane": """Composición y tamaño de hogares DANE 2005/2018 — tabla pre-agregada por municipio (3 departamentos).
Columnas: departamento, año_censo ('2005'/'2018' STRING), codigo_municipio, area_geografica,
total_hogares, promedio_personas_hogar, promedio_cuartos, hogares_unipersonales, hogares_5_o_mas.
⚠️ SIEMPRE usa SUM(total_hogares), NUNCA COUNT(*). Columna `año_censo` requiere backticks.
TOTALES EXACTOS (hogares):
  Caldas    2005: 244.685 | 2018: 309.837
  Quindío   2005: 142.982 | 2018: 174.475
  Risaralda 2005: 230.532 | 2018: 278.133""",

    "gold.jefes_hogar_ecv": """Perfil de jefes/as de hogar ECV 2025 — tabla pre-agregada (muestra representativa, 3 departamentos).
Columnas: departamento, sexo_nacer ('Hombre'/'Mujer'), estado_civil, total_jefes, edad_promedio.
⚠️ SIEMPRE usa SUM(total_jefes), NUNCA COUNT(*).
TOTALES EXACTOS (muestra):
  Caldas:    2.740 hogares | 1.571 hombres / 1.169 mujeres (42.7% jefas)
  Quindío:   2.791 hogares | 1.605 hombres / 1.186 mujeres (42.5% jefas)
  Risaralda: 2.828 hogares | 1.619 hombres / 1.209 mujeres (42.7% jefas)
La ECV es muestra probabilística — NO comparable en volúmenes absolutos con DANE (310K hogares) o SISBEN (253K).""",

    "gold.fuerza_trabajo_ecv": """Participación laboral jefes de hogar ECV 2025 — tabla pre-agregada.
Columnas: departamento, actividad_semana_pasada, posicion_ocupacional, total_personas.
⚠️ SIEMPRE usa SUM(total_personas), NUNCA COUNT(*).""",

    "gold.sivigila_intsui": """Casos de intento de suicidio SIVIGILA 2018 y 2024 — tabla pre-agregada (3 departamentos).
Columnas: departamento, año ('2018'/'2024' STRING en Gold), municipio_residencia, sexo ('Femenino'/'Masculino'),
area_geografica, pertenencia_etnica, tipo_seguridad_social, fue_hospitalizado,
codigo_subgrupo (método más frecuente: 'Intoxicación por medicamentos'),
codigo_evento, total_casos, edad_promedio, edad_min, edad_max.
⚠️ SIEMPRE usa SUM(total_casos), NUNCA COUNT(*).
TOTALES EXACTOS:
  Caldas:    2018=1.007 casos (Femenino=648, Masculino=359) | 2024=1.231 casos (F=729, M=502)
  Quindío:   2018=570 casos (F=335, M=235)                  | 2024=493 casos (F=310, M=183)
  Risaralda: 2018=793 casos (F=521, M=272)                  | 2024=1.260 casos (F=832, M=428)""",

    "gold.sivigila_vigsalpub": """Casos de violencia de género e intrafamiliar SIVIGILA 2018 y 2024 — tabla pre-agregada (3 departamentos).
Columnas: departamento, año ('2018'/'2024' STRING en Gold), municipio_residencia, sexo ('Femenino'/'Masculino'),
area_geografica, pertenencia_etnica, tipo_seguridad_social, fue_hospitalizado, condicion_final,
codigo_subgrupo (tipos: 'Violencia física'/'Violencia sexual'/'Violencia económica / patrimonial'/
'Acoso sexual'/'Abuso sexual'/'Violencia contra niños y adolescentes'/etc.),
codigo_evento, total_casos, edad_promedio, edad_min, edad_max.
⚠️ SIEMPRE usa SUM(total_casos), NUNCA COUNT(*).
TOTALES EXACTOS:
  Caldas:    2018=2.561 (F=2.128, M=433) | 2024=3.324 (F=2.623, M=701)
  Quindío:   2018=1.950 (F=1.587, M=363) | 2024=2.687 (F=2.143, M=544)
  Risaralda: 2018=2.975 (F=2.150, M=825) | 2024=3.718 (F=2.712, M=1.006)""",

    "gold.sisben_municipio": """Conteo de hogares SISBEN IV 2026 por municipio — tabla pre-agregada (3 departamentos).
Columnas: departamento, nombre_municipio, clase_territorio ('Cabecera'/'Rural disperso'/'Centro poblado'), total_hogares.
⚠️ SIEMPRE usa SUM(total_hogares), NUNCA COUNT(*). NO tiene desglose por sexo del jefe — usa bronze para eso.
TOTALES EXACTOS (todas las clases de territorio):
  Caldas:    253.243 hogares | 55.4% jefas
  Quindío:   171.903 hogares | 59.4% jefas
  Risaralda: 289.815 hogares | 61.2% jefas""",

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

    "silver.ecv_craccompohog": """ECV 2025 — características del hogar y su jefe/a (3 departamentos, 1 fila = 1 hogar).
⚠️ TODOS los registros son jefes de hogar (ORDEN=1 siempre, parentesco='Jefe/a del hogar' siempre).
Columnas clave:
  - sexo_nacer: 'Hombre' / 'Mujer' (sexo del jefe/a)
  - DIRECTORIO: ID único del hogar
  - departamento: 'Caldas' / 'Quindío' / 'Risaralda'
  - estado_civil, nivel_educativo, edad del jefe
JOIN con otros módulos ECV: ON DIRECTORIO
TOTALES EXACTOS (todos jefes de hogar):
  Caldas:    2.740 hogares | 1.571 hombres (57.3%) / 1.169 mujeres (42.7%)
  Quindío:   2.791 hogares | 1.605 hombres (57.5%) / 1.186 mujeres (42.5%)
  Risaralda: 2.828 hogares | 1.619 hombres (57.2%) / 1.209 mujeres (42.7%)""",

    "silver.ecv_fuertra": "ECV 2025 — fuerza de trabajo del jefe de hogar (2.740 registros, mismo DIRECTORIO que craccompohog). Columnas: DIRECTORIO, actividad_semana_pasada, posicion_ocupacional, horas_trabajadas, ingresos_mes_pasado.",
    "silver.ecv_salud": "ECV 2025 — salud del jefe de hogar. Columnas: DIRECTORIO, afiliado_salud, regimen_salud.",
    "silver.ecv_educacion": "ECV 2025 — educación. Columnas: DIRECTORIO, sabe_leer_escribir, nivel_educativo.",
    "silver.ecv_condvidhog": """ECV 2025 — condiciones de vida, pobreza subjetiva e ingresos (3 departamentos, 1 fila = 1 hogar).
JOIN con craccompohog ON DIRECTORIO. Totales: Caldas=2.736, Quindío=2.791, Risaralda=2.828 hogares.
COLUMNAS CLAVE DE POBREZA Y BIENESTAR:
  - se_considera_pobre: 'Sí' / 'No'  ← POBREZA SUBJETIVA
    ⚠️ Antes de filtrar ejecuta: SELECT DISTINCT se_considera_pobre FROM silver.ecv_condvidhog LIMIT 5
    (puede haber variación de encoding entre 'Sí' y 'Si' en Databricks)
  - situacion_ingresos_hogar (3 valores exactos):
      'Alcanzan para cubrir gastos mínimos'     → Caldas=73.0% / Quindío=60.8% / Risaralda=69.1%
      'No alcanzan para cubrir gastos mínimos'  → Caldas=15.6% / Quindío=28.2% / Risaralda=23.3%
      'Cubren más que los gastos mínimos'       → Caldas=11.5% / Quindío=11.0% / Risaralda=7.5%
    ⚠️ Verificar encoding con SELECT DISTINCT situacion_ingresos_hogar antes de filtrar
  - recibe_subsidio_gobierno: 'Sí' / 'No'
  - subsidio_colombia_mayor / subsidio_renta_ciudadana_hambre / subsidio_renta_ciudadana_iva / subsidio_otro
  - percepcion_economia_hogar_vs_hace_12m: percepción cambio económico vs hace 12 meses
  - eventos_adversos_hogar: si el hogar sufrió eventos negativos
  - evento_jefe_perdio_empleo / evento_cierre_negocio / evento_atraso_vivienda / etc.
  - alim_salto_comida / alim_comio_menos / alim_hogar_sin_alimentos / alim_tuvo_hambre_sin_comer
  - victima_hecho_delictivo / problemas_barrio
  - departamento: columna para filtrar por departamento""",
    "silver.ecv_servhog": "ECV 2025 — servicios del hogar (agua, energía, gas, internet). Columnas: DIRECTORIO, tipo_agua, tiene_internet, tiene_gas.",

    "silver.sivigila_intsui": """Intento de suicidio individual SIVIGILA 2018 y 2024 (3 departamentos, ~5.4K casos totales).
Columna `año` (INT): 2018 o 2024 (sin comillas). En Gold `año` es STRING.
Columnas clave:
  - sexo: 'Femenino' / 'Masculino' (NO 'Hombre'/'Mujer')
  - departamento_residencia: 'Caldas' / 'Quindío' / 'Risaralda' ← filtrar por aquí
  - municipio_ocurrencia: nombre del municipio
  - codigo_subgrupo: método más frecuente = 'Intoxicación por medicamentos'
  - edad, area_geografica, pertenencia_etnica, gp_discapacidad, fue_hospitalizado
  - Solo 2024: estrato_socioeconomico, gp_migrante, nacionalidad
TOTALES EXACTOS:
  Caldas:    2018=1.007 (F=648/M=359) | 2024=1.231 (F=729/M=502)
  Quindío:   2018=570  (F=335/M=235) | 2024=493  (F=310/M=183)
  Risaralda: 2018=793  (F=521/M=272) | 2024=1.260 (F=832/M=428)""",

    "silver.sivigila_vigsalpub": """Violencia de género e intrafamiliar individual SIVIGILA 2018 y 2024 (3 departamentos, ~17.2K casos).
Columna `año` (INT): 2018 o 2024 (sin comillas). En Gold `año` es STRING.
Columnas clave:
  - sexo: 'Femenino' / 'Masculino' (NO 'Hombre'/'Mujer')
  - codigo_departamento_ocurrencia: 'Caldas' / 'Quindío' / 'Risaralda' ← filtrar por aquí
    (¡DIFERENTE a intsui que usa departamento_residencia!)
  - municipio_ocurrencia, codigo_municipio_residencia: texto
  - codigo_subgrupo: 'Violencia física' / 'Violencia sexual' /
    'Violencia económica / patrimonial' / 'Acoso sexual' / 'Abuso sexual' /
    'Violencia contra niños y adolescentes' / etc.
  - condicion_final, edad, area_geografica, pertenencia_etnica, fue_hospitalizado
  - Solo 2024: estrato_socioeconomico, gp_migrante, gp_desmovilizado, nacionalidad
TOTALES EXACTOS:
  Caldas:    2018=2.561 (F=2.128/M=433) | 2024=3.324 (F=2.623/M=701)
  Quindío:   2018=1.950 (F=1.587/M=363) | 2024=2.687 (F=2.143/M=544)
  Risaralda: 2018=2.975 (F=2.150/M=825) | 2024=3.718 (F=2.712/M=1.006)""",

    # ── BRONZE ───────────────────────────────────────────────────────────────
    "bronze.sisben_caldas": """SISBEN IV Caldas — persona-nivel (605.843 personas, 253.243 hogares, 124 columnas).
Columnas jefe y sexo: Jefe_UG (1.0=jefe, NULL=no jefe) | sexo_persona (1=Hombre, 2=Mujer — NUMÉRICO).
CLASIFICACIÓN DE POBREZA (columnas exclusivas de Caldas):
  - Grupo: 'A'=Pobreza extrema / 'B'=Pobreza moderada / 'C'=Vulnerable / 'D'=No pobre
  - Clasificacion: 'A1'-'A5' / 'B1'-'B7' / 'C1'-'C18' / 'D1'-'D21' (nivel dentro del grupo)
TOTALES POR GRUPO (jefes hogar):
  A (extrema): 51.905 hogares (20.5%) | B (moderada): 96.882 (38.3%) | C (vulnerable): 71.822 (28.4%) | D (no pobre): 32.634 (12.9%)
Para filtrar por pobreza: WHERE Jefe_UG = 1 AND Grupo = 'A' (extrema) o Grupo IN ('A','B') (pobres)
Para jefas en pobreza extrema: WHERE Jefe_UG = 1 AND Grupo = 'A' AND sexo_persona = 2""",

    "bronze.sisben_quindio": """SISBEN IV Quindío — persona-nivel (366.319 personas, 171.903 hogares, 91 columnas).
⚠️ ESTRUCTURA DIFERENTE a bronze.sisben_caldas:
  - Columna jefe: parentesco_jefe_hogar = 'Jefe del hogar' (TEXTO)
  - Columna sexo: sexo = 'Hombre' / 'Mujer' (TEXTO)
  - Columna clasificación: clasificacion_sisben_iv ('A1'-'D21') — para el grupo extrae el primer carácter: SUBSTRING(clasificacion_sisben_iv, 1, 1)
CLASIFICACIÓN DE POBREZA:
  A (extrema): 30.838 hogares (17.9%) | B (moderada): 66.075 (38.4%) | C (vulnerable): 52.129 (30.3%) | D (no pobre): 22.861 (13.3%)
Para filtrar por pobreza: WHERE parentesco_jefe_hogar = 'Jefe del hogar' AND SUBSTRING(clasificacion_sisben_iv, 1, 1) = 'A'""",

    "bronze.sisben_risaralda": """SISBEN IV Risaralda — persona-nivel (641.615 personas, 289.815 hogares, 91 columnas).
⚠️ MISMA ESTRUCTURA que bronze.sisben_quindio (diferente a bronze.sisben_caldas):
  - Columna jefe: parentesco_jefe_hogar = 'Jefe del hogar' (TEXTO)
  - Columna sexo: sexo = 'Hombre' / 'Mujer' (TEXTO)
  - Columna clasificación: clasificacion_sisben_iv ('A1'-'D21') — grupo = SUBSTRING(clasificacion_sisben_iv, 1, 1)
CLASIFICACIÓN DE POBREZA:
  A (extrema): 52.877 hogares (18.2%) | B (moderada): 100.225 (34.6%) | C (vulnerable): 96.769 (33.4%) | D (no pobre): 39.944 (13.8%)""",

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
