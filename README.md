# Clasificación de la Potabilidad del Agua con Machine Learning

# Introducción

Este proyecto documenta el desarrollo de un modelo de Machine Learning enfocado en la clasificación de la potabilidad del agua basándose en sus propiedades fisicoquímicas. El objetivo es construir un modelo capaz de clasificar si una muestra de agua es potable o no, lo que puede ayudar a identificar fuentes de agua seguras y detectar problemas de contaminación tempranamente. El proceso siguió un enfoque iterativo, abarcando desde la definición del caso de uso, análisis exploratorio de datos, preprocesamiento, formulación de hipótesis de modelado, hasta la optimización y evaluación detallada de un modelo final.

## Autor

Andrea Pardo Gispert

## Dataset

El dataset principal utilizado es "Water Quality" de Kaggle. Contiene 3276 muestras de agua con 9 características predictoras y 1 variable objetivo (`Potability`).

Las características predictoras son:

- **ph**: Nivel de pH del agua (0 a 14)
- **Hardness**: Dureza del agua en mg/L
- **Solids**: Sólidos totales disueltos en ppm
- **Chloramines**: Cloraminas en ppm
- **Sulfate**: Sulfatos en mg/L
- **Conductivity**: Conductividad eléctrica en micros/cm
- **Organic_carbon**: Carbono orgánico total en ppm
- **Trihalomethanes**: Trihalometanos en microg/L
- **Turbidity**: Turbidez del agua en NTU
- **Potability**: Variable objetivo (1: potable, 0: no potable)

Un aspecto crucial del dataset es el desequilibrio de clases, con aproximadamente el 60% de las muestras correspondiendo a agua no potable (clase 0) y el 40% a agua potable (clase 1).

## Requisitos Técnicos y Métricas de Evaluación

**Entorno de Ejecución:** Google Colab o máquina local con CPU y 4-8 GB de RAM.
**Tiempo de Entrenamiento:** No debe exceder los 5 minutos para modelos individuales.
**Librerías Principales:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `openml`.

**Métricas de Evaluación:**

- **Accuracy (Exactitud)**: Proporción de predicciones correctas.
- **Precisión**: Proporción de verdaderos positivos sobre el total de predicciones positivas (importante para evitar falsos positivos).
- **Recall (Sensibilidad)**: Proporción de verdaderos positivos sobre el total de elementos que son realmente positivos (crucial para identificar todas las muestras potables).
- **F1-Score**: Media armónica de Precisión y Recall, útil en desequilibrio de clases.
- **AUC-ROC**: Mide la capacidad del clasificador para distinguir entre clases, robusta ante el desequilibrio de clases.

**Umbrales de Rendimiento:**

- **Precisión mínima aceptable**: Accuracy y Precisión > 70-75%.
- **Precisión deseable**: Accuracy y Precisión > 85%.

## Preprocesamiento de Datos

Se realizó un Análisis Exploratorio de Datos (EDA) exhaustivo.

- **Calidad de Datos**: Se identificaron valores nulos en 'ph', 'Sulfate' y 'Trihalomethanes', los cuales fueron imputados utilizando la mediana de cada columna. Los outliers se mantuvieron para observar cómo los algoritmos los manejaban.
- **Análisis de Características**: Se exploraron las distribuciones y su relación con la potabilidad. Las características numéricas se escalaron utilizando `StandardScaler`.
- **División del Dataset**: El dataset se dividió en conjuntos de entrenamiento (70%) y prueba (30%) utilizando `train_test_split` con estratificación para mantener la proporción de clases. Se usó `random_state=42` para reproducibilidad.

## Modelado y Resultados

Se exploraron varios modelos en un enfoque iterativo:

### 1. Modelo Base: Regresión Logística

- **Resultados**:
    - Accuracy: 0.5229
    - Precisión (Clase 1 - Potable): 0.4154
    - Recall (Clase 1 - Potable): 0.5509
    - F1-Score (Clase 1 - Potable): 0.4736
    - AUC-ROC: 0.5295
- **Análisis**: El modelo mostró limitaciones para el problema complejo y desequilibrado, aunque el uso de `class_weight='balanced'` mitigó el sesgo hacia la clase mayoritaria.

### 2. Experimento con Ingeniería de Características (ph*Sulfate)

- **Resultados (Regresión Logística con FE)**:
    - Accuracy: 0.6256
    - Precisión (Clase 1 - Potable): 0.6056
    - Recall (Clase 1 - Potable): 0.1123
    - F1-Score (Clase 1 - Potable): 0.1894
    - AUC-ROC: 0.6050
- **Análisis**: Aunque mejoró Accuracy, Precisión y AUC-ROC, el Recall disminuyó drásticamente, lo que indica que la característica de interacción no aportó valor predictivo positivo para este modelo lineal. No se usó en iteraciones posteriores.

### 3. Modelos de Ensemble: Random Forest Classifier

### Random Forest Inicial (Modelo_v2)

- **Resultados**:
    - Accuracy: 0.6450
    - Precisión (Clase 1): 0.6012
    - Recall (Clase 1): 0.2637
    - F1-Score (Clase 1): 0.3666
    - AUC-ROC: 0.6604
- **Análisis**: Mejora significativa sobre la Regresión Logística, pero el Recall seguía siendo bajo.

### Random Forest Optimizado (Modelo_v3 - GridSearchCV)

- **Resultados**:
    - Accuracy: 0.6562
    - Precisión (Clase 1): 0.6077
    - Recall (Clase 1): 0.3316
    - F1-Score (Clase 1): 0.4291
    - AUC-ROC: 0.6656
- **Análisis**: Mejora en todas las métricas, con la Precisión más alta hasta el momento. Sin embargo, el Recall aún era relativamente bajo.

### 4. Modelos de Boosting: LightGBM Classifier

### LightGBM Inicial (Modelo_v4)

- **Resultados**:
    - Accuracy: 0.6429
    - Precisión (Clase 1): 0.5476
    - Recall (Clase 1): 0.4804
    - F1-Score (Clase 1): 0.5118
    - AUC-ROC: 0.6496
- **Análisis**: F1-Score muy prometedor y mejor Recall comparado con Random Forest, posicionándolo como un fuerte candidato.

### LightGBM Optimizado (Modelo_v5 - GridSearchCV)

- **Parámetros**: `{'is_unbalance': True, 'learning_rate': 0.05, 'max_depth': -1, 'min_child_samples': 20, 'n_estimators': 100, 'num_leaves': 31, 'objective': 'binary', 'random_state': 42}`
- **Resultados**:
    - Accuracy: 0.6399
    - Precisión (Clase 1): 0.5428
    - Recall (Clase 1): 0.4804
    - F1-Score (Clase 1): 0.5097
    - AUC-ROC: 0.6610
- **Análisis**: Este modelo demostró ser el más equilibrado en sus predicciones, minimizando el impacto del desequilibrio de clases de forma efectiva y ofreciendo el mejor balance entre Precisión y Recall hasta el momento.

### 5. Refinamiento Adicional

### Manejo del Desequilibrio con SMOTE (Modelo_v6 - LightGBM Optimizado + SMOTE)

- **Resultados**:
    - Accuracy: 0.6124
    - Precisión (Clase 1): 0.5027
    - Recall (Clase 1): 0.4935
    - F1-Score (Clase 1): 0.4980
    - AUC-ROC: 0.6491
- **Análisis**: Aunque el Recall mejoró ligeramente, se redujeron la Precisión, Accuracy y F1-Score general, por lo que no se consideró beneficioso.

### Reintroducción de Ingeniería de Características (Modelo_v7 - LightGBM Optimizado + FE)

- **Resultados**:
    - Accuracy: 0.6328
    - Precisión (Clase 1): 0.5312
    - Recall (Clase 1): 0.4883
    - F1-Score (Clase 1): 0.5088
    - AUC-ROC: 0.6488
- **Análisis**: No mostró una mejora sustancial en comparación con el LightGBM optimizado sin esta característica.

## Conclusión

Este proyecto demostró la efectividad de un enfoque iterativo en Machine Learning para enfrentar un problema del mundo real. Aunque no se alcanzaron las métricas aspiracionales más altas (Accuracy y Precisión > 70-75%), se logró desarrollar un clasificador robusto y con el mejor rendimiento posible dadas las características del dataset actual. La Accuracy obtenida (0.6399) es sustancialmente mejor que una clasificación aleatoria, y el AUC-ROC de 0.6610 es un indicador sólido de la capacidad discriminatoria del modelo en un problema con desequilibrio de clases.

El aprendizaje obtenido sobre la importancia del preprocesamiento, la evaluación de métricas múltiples y la validación de hipótesis es de gran valor.

## Futuro Trabajo

- Integración de fuentes de datos adicionales.
- Exploración de técnicas de ingeniería de características más avanzadas.
- Evaluación de modelos de red neuronal para problemas con patrones más complejos
