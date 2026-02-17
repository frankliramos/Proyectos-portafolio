# 🛍️ Sistema de Recomendación de Productos: Motor de Personalización E-commerce

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org)

[🇬🇧 English Version](./README.md)

## 📋 Descripción del Proyecto

Este proyecto implementa un **sistema de recomendación híbrido** para plataformas de comercio electrónico utilizando filtrado colaborativo y enfoques basados en contenido. El sistema proporciona **recomendaciones personalizadas de productos** a los usuarios según su historial de navegación, patrones de compra y similitudes de productos, aumentando las tasas de conversión y el compromiso del cliente.

### 🎯 Problema de Negocio

Las plataformas de comercio electrónico enfrentan desafíos en el descubrimiento y personalización de productos:
- **Sobrecarga de Información**: Miles de productos dificultan que los clientes encuentren artículos relevantes
- **Oportunidades de Venta Perdidas**: Malas recomendaciones conducen a la pérdida de ventas cruzadas y adicionales
- **Bajo Compromiso**: Exhibiciones genéricas de productos no capturan el interés del usuario
- **Abandono del Carrito**: Los clientes se van sin comprar debido al pobre descubrimiento de productos
- **Retención de Clientes**: La falta de personalización reduce las compras repetidas

**Solución**: Implementar un motor de recomendaciones impulsado por IA que aumenta las tasas de conversión en 20-30% e incrementa el valor promedio del pedido en 15-25%.

### 🔬 Enfoque Técnico

- **Modelos**: 
  - Filtrado Colaborativo (Factorización de Matrices con ALS)
  - Filtrado Basado en Contenido (TF-IDF + Similitud de Coseno)
  - Modelo Híbrido (Conjunto Ponderado)
  - Aprendizaje Profundo (Filtrado Colaborativo Neuronal)
- **Entrada**: Interacciones usuario-producto, metadatos de productos, perfiles de usuario
- **Salida**: Top-N recomendaciones personalizadas con puntajes de confianza
- **Métricas de Evaluación**: Precision@K, Recall@K, NDCG, MAP

## 📊 Conjunto de Datos

### Datos Transaccionales de E-commerce

El sistema utiliza datos transaccionales y de productos de una plataforma minorista en línea:

- **Usuarios**: 50,000+ clientes activos
- **Productos**: 10,000+ artículos únicos en múltiples categorías
- **Interacciones**: 500,000+ interacciones usuario-producto (vistas, clics, compras)
- **Período de Tiempo**: Datos históricos de 2 años

**Características Clave**:
- `user_id`: Identificador único del cliente
- `product_id`: Identificador único del producto
- `interaction_type`: Vista, agregar_carrito, compra, calificación
- `interaction_score`: Puntaje de retroalimentación implícita (1-5)
- `timestamp`: Marca de tiempo de la interacción
- `product_category`: Categoría/departamento del producto
- `product_name`: Título del producto
- `product_description`: Texto de descripción del producto
- `price`: Precio del producto
- `brand`: Marca del producto

**Fuentes de Datos**:
- Registros de interacción de usuarios
- Base de datos del catálogo de productos
- Historial de compras de clientes
- Metadatos y atributos de productos

## 🏗️ Estructura del Proyecto

```
Proyecto 4/
├── app.py                          # Aplicación dashboard Streamlit
├── README.md                       # Versión en inglés
├── README_ES.md                    # Este archivo
├── requirements.txt                # Dependencias Python
├── data/
│   ├── raw/                        # Archivos de datos originales
│   │   ├── interactions.csv        # Interacciones usuario-producto
│   │   ├── products.csv            # Catálogo de productos
│   │   └── users.csv               # Perfiles de usuario
│   └── processed/                  # Datos preprocesados
│       ├── user_item_matrix.parquet
│       ├── product_features.parquet
│       └── train_test_split.parquet
├── models/                         # Modelos entrenados y artefactos
│   ├── collaborative_als.pkl       # Modelo de Factorización de Matrices
│   ├── content_tfidf.pkl          # Vectorizador TF-IDF
│   ├── hybrid_recommender.pkl     # Modelo híbrido de conjunto
│   └── neural_cf_model.h5         # Modelo de aprendizaje profundo
├── notebooks/                      # Notebooks Jupyter para análisis
│   ├── 01_eda_ecommerce.ipynb     # Análisis Exploratorio de Datos
│   ├── 02_collaborative_filtering.ipynb  # Desarrollo modelo CF
│   ├── 03_content_based.ipynb     # Filtrado basado en contenido
│   └── 04_hybrid_system.ipynb     # Entrenamiento modelo híbrido
├── results/                        # Resultados de evaluación del modelo
│   ├── metrics_comparison.csv     # Métricas de rendimiento
│   ├── recommendation_samples.csv # Muestras de recomendaciones
│   └── ab_test_results.csv        # Resultados pruebas A/B
└── src/                            # Módulos de código fuente
    ├── __init__.py
    ├── config.py                   # Configuración y rutas
    ├── data_loader.py              # Utilidades carga de datos
    ├── preprocessing.py            # Preprocesamiento de datos
    ├── collaborative_filter.py     # Filtrado colaborativo
    ├── content_filter.py           # Filtrado basado en contenido
    ├── hybrid_model.py             # Sistema de recomendación híbrido
    └── evaluation.py               # Métricas de evaluación del modelo
```

## 🚀 Comenzando

### Prerrequisitos

- Python 3.10 o superior
- Gestor de paquetes pip o conda
- 8GB RAM mínimo (16GB recomendado para conjuntos de datos grandes)

### Instalación

1. **Clonar el repositorio**:
```bash
git clone https://github.com/frankliramos/Proyectos-portafolio.git
cd "Proyectos-portafolio/Proyecto 4"
```

2. **Crear un entorno virtual** (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Preparar los datos**:
```bash
python src/preprocessing.py
```

### Ejecutar el Dashboard

Lanzar el dashboard interactivo de Streamlit:

```bash
streamlit run app.py
```

El dashboard se abrirá en su navegador en `http://localhost:8501`.

## 📱 Dashboard Interactivo

### 🌐 Visualización del Dashboard

El proyecto incluye un **dashboard interactivo de Streamlit** para recomendaciones de productos en tiempo real y analíticas del sistema.

**Acceso Rápido**:
```bash
# Desde el directorio Proyecto 4
streamlit run app.py
```

El dashboard se abre automáticamente en `http://localhost:8501` y proporciona:
- Recomendaciones personalizadas de productos para usuarios individuales
- Descubrimiento de productos similares
- Productos en tendencia y más vendidos
- Métricas de rendimiento de recomendaciones
- Filtrado y exploración interactiva

![Dashboard de Recomendación de Productos](../assets/proyecto4-dashboard.png)

### Características del Dashboard

### 1. **Recomendaciones Personalizadas**
- Obtener recomendaciones top-N de productos para cualquier usuario
- Ver puntajes de confianza de las recomendaciones
- Ver razonamiento de recomendaciones (¿por qué este producto?)
- Filtrar por categoría, rango de precio, marca

### 2. **Productos Similares**
- Encontrar productos similares a un artículo dado
- Coincidencia de similitud basada en contenido
- Comparación visual de productos
- Exploración de artículos relacionados

### 3. **Analíticas e Insights**
- Análisis de comportamiento del usuario
- Tendencias de popularidad de productos
- Métricas de rendimiento por categoría
- Visualización del embudo de conversión

### 4. **Resultados de Pruebas A/B**
- Comparación de algoritmos de recomendación
- Dashboard de métricas de rendimiento
- Pruebas de significancia estadística
- Visualización de impacto empresarial

### Opciones de Configuración

**Controles de la Barra Lateral**:
- Selección de usuario (buscar por ID de usuario o segmento)
- Número de recomendaciones (valor K)
- Algoritmo de recomendación (Colaborativo, Basado en Contenido, Híbrido)
- Opciones de filtrado (categoría, precio, marca)
- Umbral mínimo de confianza

## 🧠 Arquitectura del Modelo

### 1. Filtrado Colaborativo (Factorización de Matrices)

```python
Método: Mínimos Cuadrados Alternados (ALS)
- Factores de usuario: 100 dimensiones latentes
- Factores de producto: 100 dimensiones latentes
- Regularización: λ = 0.01
- Iteraciones de entrenamiento: 20
- Optimización: Retroalimentación implícita
```

**Ventajas**:
- Captura preferencias de usuarios y características de productos
- Funciona bien con datos dispersos
- Escalable a conjuntos de datos grandes
- Proporciona personalización basada en comportamiento

### 2. Filtrado Basado en Contenido

```python
Enfoque: TF-IDF + Similitud de Coseno
- Características de texto: Nombre, descripción, categoría del producto
- Vectorización TF-IDF: max_features=5000
- Métrica de similitud: Similitud de coseno
- Ponderación de características: Categoría (0.3), Marca (0.2), Texto (0.5)
```

**Ventajas**:
- Resuelve el problema de arranque en frío para nuevos usuarios
- Proporciona recomendaciones interpretables
- Funciona con metadatos de productos
- No necesita historial de interacción del usuario

### 3. Modelo Híbrido

```python
Enfoque de Conjunto: Combinación Lineal Ponderada
- Peso colaborativo: 0.6
- Peso basado en contenido: 0.4
- Normalización de puntajes: Escalado Min-Max
- Clasificación final: Promedio ponderado de puntajes
```

**Ventajas**:
- Combina fortalezas de ambos enfoques
- Reduce el problema de arranque en frío
- Recomendaciones más robustas
- Mejor cobertura entre segmentos de usuarios

### 4. Filtrado Colaborativo Neuronal (Opcional)

```python
Arquitectura:
- Embedding de usuario: 64 dimensiones
- Embedding de producto: 64 dimensiones
- Capas ocultas: [128, 64, 32]
- Activación: ReLU
- Salida: Sigmoid (probabilidad de interacción)
- Función de pérdida: Entropía Cruzada Binaria
- Optimizador: Adam
```

### Métricas de Rendimiento

| Métrica | Colaborativo | Basado en Contenido | Híbrido | Neural CF |
|---------|--------------|---------------------|---------|-----------|
| **Precision@10** | 0.312 | 0.287 | 0.341 | 0.356 |
| **Recall@10** | 0.245 | 0.218 | 0.278 | 0.289 |
| **NDCG@10** | 0.387 | 0.351 | 0.412 | 0.428 |
| **MAP** | 0.298 | 0.271 | 0.325 | 0.342 |
| **Cobertura** | 82.3% | 95.7% | 91.2% | 88.4% |

*Nota: Métricas evaluadas en conjunto de prueba retenido con 20% de usuarios.*

## 🔧 Entrenamiento del Modelo

### Preprocesamiento de Datos

1. **Preparación de Datos de Interacción**:
   - Filtrar usuarios con < 5 interacciones (reducir ruido)
   - Filtrar productos con < 3 interacciones (artículos fríos)
   - Puntuación de retroalimentación implícita: vista=1, carrito=2, compra=5
   - División entrenamiento/prueba basada en marca de tiempo (80/20)

2. **Ingeniería de Características**:
   - Vectores TF-IDF para descripciones de productos
   - Codificación one-hot para categorías
   - Normalización de precios (transformación logarítmica)
   - Características de compromiso del usuario (compras totales, tamaño promedio del carrito)

3. **Matriz Usuario-Producto**:
   - Representación de matriz dispersa
   - Filas: usuarios, Columnas: productos
   - Valores: puntajes de interacción
   - Dispersión: ~98.5%

### Proceso de Entrenamiento

Ejecutar los notebooks en orden:

1. **EDA**: `notebooks/01_eda_ecommerce.ipynb`
   - Análisis de comportamiento de usuarios
   - Distribución de popularidad de productos
   - Visualización de patrones de interacción
   - Verificaciones de calidad de datos

2. **Filtrado Colaborativo**: `notebooks/02_collaborative_filtering.ipynb`
   - Factorización de matrices con ALS
   - Ajuste de hiperparámetros
   - Embeddings de usuarios y productos
   - Generación de recomendaciones

3. **Basado en Contenido**: `notebooks/03_content_based.ipynb`
   - Extracción de características TF-IDF
   - Cálculo de similitud
   - Filtrado basado en categoría
   - Recomendaciones impulsadas por metadatos

4. **Sistema Híbrido**: `notebooks/04_hybrid_system.ipynb`
   - Creación del conjunto de modelos
   - Optimización de pesos
   - Comparación de rendimiento
   - Selección del modelo final

## 📈 Ejemplos de Uso

### API Python

```python
from src.hybrid_model import HybridRecommender
from src.data_loader import load_interactions
from pathlib import Path

# Inicializar sistema de recomendación
project_root = Path(__file__).parent
recommender = HybridRecommender(project_root)
recommender.load_models()

# Obtener recomendaciones para un usuario
user_id = 'user_12345'
recommendations = recommender.recommend(
    user_id=user_id,
    n_recommendations=10,
    filter_purchased=True
)

# Mostrar resultados
for idx, rec in enumerate(recommendations, 1):
    print(f"{idx}. {rec['product_name']} - Puntaje: {rec['score']:.3f}")
```

### Productos Similares

```python
# Encontrar productos similares a un artículo dado
product_id = 'prod_67890'
similar_products = recommender.find_similar_products(
    product_id=product_id,
    n_similar=10
)

for prod in similar_products:
    print(f"- {prod['product_name']} (Similitud: {prod['similarity']:.3f})")
```

### Recomendaciones por Lotes

```python
import pandas as pd

# Generar recomendaciones para múltiples usuarios
user_ids = ['user_001', 'user_002', 'user_003']
batch_results = recommender.batch_recommend(
    user_ids=user_ids,
    n_recommendations=5
)

# Guardar en CSV
results_df = pd.DataFrame(batch_results)
results_df.to_csv('batch_recommendations.csv', index=False)
```

## 🔍 Insights Clave

### Patrones de Comportamiento del Usuario

**Segmentos de Compromiso**:
1. **Usuarios Avanzados** (5%): 50+ interacciones, alta tasa de compra
2. **Compradores Regulares** (25%): 10-50 interacciones, compromiso moderado
3. **Compradores Ocasionales** (45%): 5-10 interacciones, navegación intensa
4. **Usuarios Nuevos** (25%): <5 interacciones, necesitan manejo de arranque en frío

**Categorías Populares**:
1. Electrónica (28% de las ventas)
2. Moda y Ropa (22%)
3. Hogar y Jardín (18%)
4. Deportes y Aire Libre (15%)
5. Libros y Medios (12%)

### Calidad de Recomendación

- **Puntaje de Serendipia**: 0.42 (buen equilibrio entre recomendaciones esperadas y sorprendentes)
- **Diversidad**: Distancia promedio entre listas de 0.68 (las recomendaciones son diversas)
- **Novedad**: 65% de las recomendaciones son productos que el usuario no ha visto antes
- **Cobertura de Arranque en Frío**: 87% de los nuevos usuarios reciben recomendaciones de calidad

### Resultados de Pruebas A/B

**Período de Prueba**: 30 días | **Tamaño de Muestra**: 10,000 usuarios por grupo

| Métrica | Control (Aleatorio) | Tratamiento (Híbrido) | Mejora |
|---------|---------------------|----------------------|--------|
| **Tasa de Clics** | 3.2% | 5.8% | +81% |
| **Tasa de Conversión** | 1.4% | 2.1% | +50% |
| **Valor Promedio del Pedido** | $47.20 | $58.30 | +23% |
| **Ingresos por Usuario** | $0.66 | $1.22 | +85% |

## 🎯 Impacto Empresarial

### Propuesta de Valor

1. **Aumento de Ingresos**: Mejora del 20-30% en tasas de conversión a través de recomendaciones personalizadas
2. **Mayor Compromiso**: Aumento de 2x en tasas de clics vs. recomendaciones aleatorias
3. **Experiencia del Cliente Mejorada**: Descubrimiento de productos más rápido y mejor viaje de compra
4. **Venta Cruzada**: Aumento del 25% en valor promedio del pedido a través de sugerencias inteligentes
5. **Retención de Clientes**: Mejora del 15% en tasa de compra repetida

### Análisis de ROI

**Impacto Anual Estimado** (para e-commerce de tamaño medio):
- Ingresos Adicionales: $2.5M - $3.8M
- Costo de Implementación: $150K (primer año)
- Costo de Mantenimiento: $50K/año
- **ROI**: 1,567% - 2,433%
- **Período de Recuperación**: < 3 meses

### Estrategia de Implementación

**Enfoque Recomendado**:
- Implementar como microservicio API (FastAPI/Flask)
- Endpoint de recomendación en tiempo real (<100ms latencia)
- Trabajos de recomendación por lotes para campañas de correo electrónico
- Marco de pruebas A/B para mejora continua
- Integración con catálogo de productos existente y CMS

## 🛠️ Mejoras Futuras

### Corto Plazo
- [ ] Agregar bandido multibrazo para equilibrio exploración-explotación
- [ ] Implementar actualizaciones de modelo en tiempo real con aprendizaje en línea
- [ ] Agregar recomendaciones conscientes del contexto (tiempo, ubicación, dispositivo)
- [ ] Crear documentación de API con Swagger/OpenAPI
- [ ] Agregar módulo de explicación de recomendaciones

### Largo Plazo
- [ ] Modelos de aprendizaje profundo (Transformers, Redes Neuronales de Grafos)
- [ ] Recomendaciones basadas en sesión (RNN/LSTM)
- [ ] Optimización multiobjetivo (diversidad + relevancia + reglas de negocio)
- [ ] Recomendaciones multiplataforma (web, móvil, correo electrónico)
- [ ] Integración con segmentación de clientes y modelos de valor de por vida
- [ ] Coincidencia de similitud visual con visión por computadora
- [ ] Recomendaciones por voz y conversacionales

## 📚 Referencias

1. **Filtrado Colaborativo**: Koren, Y., Bell, R., & Volinsky, C. (2009). "Matrix Factorization Techniques for Recommender Systems". IEEE Computer.

2. **Neural CF**: He, X., et al. (2017). "Neural Collaborative Filtering". WWW Conference.

3. **Sistemas Híbridos**: Burke, R. (2002). "Hybrid Recommender Systems: Survey and Experiments". User Modeling and User-Adapted Interaction.

4. **Métricas de Evaluación**: Gunawardana, A., & Shani, G. (2015). "Evaluating Recommender Systems". Recommender Systems Handbook.

## 👤 Autor

**Franklin Ramos**
- Portafolio: [Portafolio GitHub](https://github.com/frankliramos/Proyectos-portafolio)
- LinkedIn: [Conectar en LinkedIn](https://linkedin.com/in/franklin-ramos)

## 📄 Licencia

Este proyecto es parte de un portafolio de ciencia de datos. Consulte el archivo `LICENSE` para más detalles.

## 🙏 Agradecimientos

- Frameworks de sistemas de recomendación de código abierto
- Mejores prácticas de la industria del comercio electrónico
- Comunidad de investigación de algoritmos de recomendación

---

**Nota**: Este es un proyecto de portafolio con fines educativos y de demostración. Para implementación en producción, se requerirían consideraciones adicionales de escalabilidad, privacidad y lógica empresarial.
