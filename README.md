# PROYECTO_METODOLOGIAS_AGILES

## 📊 Dashboard de Análisis de Trámites SUIT

Este proyecto implementa un dashboard interactivo para el análisis y visualización de datos del Sistema Único de Información de Trámites (SUIT) de Colombia, utilizando metodologías ágiles de desarrollo.

## 🎯 Objetivo

El proyecto tiene como objetivo crear una herramienta de análisis de datos que permita visualizar y analizar los trámites registrados en el SUIT, proporcionando insights sobre la evolución temporal y distribución geográfica de los trámites en Colombia.

## 🏗️ Arquitectura del Proyecto

### Componentes Principales

1. **Script de Preprocesamiento** (`Proyecto_Metodologías_Ágiles.ipynb`)
   - Descarga datos del dataset SUIT desde datos.gov.co
   - Procesa y limpia los datos
   - Genera archivo Parquet optimizado

2. **Dashboard Interactivo** (`dashboard.py`)
   - Aplicación web desarrollada con Streamlit
   - Visualizaciones interactivas con Altair y PyDeck
   - Filtros dinámicos en cascada

## 📈 Funcionalidades

### 🔍 Análisis Temporal
- **Evolución mensual por año**: Gráfico de líneas superpuestas mostrando tendencias anuales
- **Indicadores clave**: Total de trámites, promedio mensual, variación porcentual entre años
- **Filtros temporales**: Selección por años y meses con filtros dependientes

### 🗺️ Análisis Geográfico
- **Mapa interactivo**: Visualización de distribución geográfica con PyDeck
- **Filtros geográficos**: Por departamento, municipio y entidad
- **Coordenadas validadas**: Verificación de coordenadas geográficas válidas

### 📊 Visualizaciones
- **KPIs principales**: Métricas clave en tiempo real
- **Gráfico de evolución**: Serie temporal interactiva
- **Mapa de calor geográfico**: Distribución espacial de trámites
- **Tabla resumen**: Datos agregados por año/mes

## 🛠️ Tecnologías Utilizadas

### Backend y Procesamiento
- **Python 3.10+**
- **Pandas**: Manipulación y análisis de datos
- **Sodapy**: Cliente para API de datos.gov.co
- **NumPy**: Operaciones numéricas

### Frontend y Visualización
- **Streamlit**: Framework web para dashboards
- **Altair**: Visualizaciones estadísticas interactivas
- **PyDeck**: Mapas interactivos 3D
- **Parquet**: Formato de almacenamiento optimizado

### Herramientas de Desarrollo
- **Jupyter Notebook**: Desarrollo y prototipado
- **Localtunnel**: Exposición local para testing
- **Git**: Control de versiones

## 📊 Dataset

### Fuente de Datos
- **Origen**: Sistema Único de Información de Trámites (SUIT)
- **Plataforma**: datos.gov.co
- **Dataset ID**: 48fq-mxnm
- **Registros**: ~536,000 trámites

### Campos Principales
- `fecha_de_actualización`: Fecha de actualización del trámite
- `departamento`: Departamento donde se realizó el trámite
- `municipio`: Municipio específico
- `nombre_de_la_entidad`: Entidad responsable del trámite
- `latitud_municipio` / `longitud_municipio`: Coordenadas geográficas

## 🚀 Instalación y Uso

### Prerrequisitos
```bash
pip install pandas sodapy altair pydeck streamlit
npm install localtunnel
```

### Ejecución

1. **Preprocesamiento de datos**:
   ```bash
   # Ejecutar el notebook para descargar y procesar datos
   jupyter notebook Proyecto_Metodologías_Ágiles.ipynb
   ```

2. **Ejecutar dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

3. **Acceso remoto** (opcional):
   ```bash
   npx localtunnel --port 8501
   ```

## 📁 Estructura del Proyecto

```
PROYECTO_METODOLOGIAS_AGILES/
├── Proyecto_Metodologías_Ágiles.ipynb  # Script de preprocesamiento
├── dashboard.py                         # Dashboard principal
├── data/
│   └── suit_tramites.parquet           # Datos procesados
├── README.md                           # Documentación
└── package.json                        # Configuración Node.js
```

## 🔧 Características Técnicas

### Optimizaciones
- **Caché inteligente**: `@st.cache_data` con TTL de 1 hora
- **Filtros en cascada**: Optimización de consultas dependientes
- **Formato Parquet**: Compresión y acceso rápido a datos
- **Validación de coordenadas**: Filtrado de datos geográficos válidos

### Manejo de Datos
- **Limpieza automática**: Conversión de tipos y manejo de errores
- **Columnas derivadas**: Año, mes, fecha_mes, coordenadas válidas
- **Filtros dinámicos**: Actualización automática de opciones disponibles

## 📈 Métricas y KPIs

- **Total de trámites**: Conteo total según filtros aplicados
- **Promedio mensual**: Promedio de trámites por mes
- **Variación porcentual**: Cambio entre primer y último año seleccionado
- **Distribución geográfica**: Concentración por departamento/municipio

## 🎨 Interfaz de Usuario

### Sidebar de Filtros
- **Años**: Selección múltiple de años disponibles
- **Meses**: Filtro dependiente de años seleccionados
- **Departamento**: Filtro geográfico por departamento
- **Municipio**: Filtro dependiente del departamento
- **Entidad**: Filtro por entidad responsable (limitado a 300 opciones)

### Visualizaciones Principales
- **Gráfico de evolución**: Líneas superpuestas por año
- **Mapa interactivo**: Scatterplot con radio proporcional a cantidad
- **Tabla expandible**: Resumen por año/mes

## 🔄 Flujo de Datos

1. **Extracción**: Descarga desde API de datos.gov.co
2. **Transformación**: Limpieza y creación de columnas derivadas
3. **Carga**: Almacenamiento en formato Parquet
4. **Visualización**: Dashboard interactivo con filtros dinámicos

## 📝 Metodologías Ágiles

Este proyecto implementa principios de desarrollo ágil:
- **Iteraciones cortas**: Desarrollo incremental
- **Feedback continuo**: Testing y validación constante
- **Documentación viva**: README actualizado con el código
- **Flexibilidad**: Adaptación a cambios en requerimientos

## 🤝 Contribución

Para contribuir al proyecto:
1. Fork del repositorio
2. Crear rama de feature
3. Implementar cambios
4. Testing local
5. Pull request con descripción detallada

## 📄 Licencia

Este proyecto es parte de un trabajo académico sobre metodologías ágiles en análisis de datos.

---

**Desarrollado como parte del proyecto de Metodologías Ágiles**  
*Análisis de datos del Sistema Único de Información de Trámites (SUIT) de Colombia*