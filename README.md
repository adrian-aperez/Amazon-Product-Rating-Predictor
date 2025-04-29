# Product Rating Predictor

## Descripción

**Product Rating Predictor** es una aplicación diseñada para predecir las calificaciones de productos basándose en datos de productos de Amazon y las características del producto (En este caso se elaboró con datos de productos de higiene masculina). 

Utiliza algoritmos de aprendizaje automático para generar predicciones separados en dos modelos distintos. Modelo_bajo está entrenado con productos con menos de 60 reviews y Modelo_alto está entrenado con productos con más de 60 reviews.


## Resumen del Proyecto


1.Extracción de datos de fuentes externas y la limpieza de los mismos.

2.Análisis exploratorio de los datos (EDA).

3.Feature Engineering:

 1.   Elaboración de funciones y preprocesador como transformadores de los datos.

 2.   Utilización de TF-IDF (Term Frequency-Inverse Document Frequency) como herramienta de NLP para procesar texto.

4.Utilización de algoritmos tanto de Machine Learning como de Deep Learning.

5.Hiperparametrización de los algoritmos escogidos y validación cruzada para optimización de los resultados.
   
6.Puesta en producción usando la herramienta Streamlit.



## Características
| Característica     | Descripción                                         |
|--------------------|-----------------------------------------------------|
| 🌐 Responsivo     | Compatible con todos los tamaños de pantalla.        |
| ⚡ Rápido         | Carga optimizada y procesos ágiles.                  |
| ⭐​ Predictivo     | Predicción de calificaciones de productos de Amazon. |
| 📊​ Analítico      | Análisis exploratorio con insights de negocio.       |
| 📈​ Integración    | Integración con datos y características reales.      |
| 💻 Soporte        | Soporta múltiples modelos de aprendizaje automático. |
| 🚀​ Aplicación     | Contiene aplicación visual en Streamlit.             |

## Herramientas
- Python        
- Scikit-learn
- Pandas        
- Numpy
- Pycaret
- Joblib        
- NLTK
- Seaborn       
- Matplotlib


## Instalación

1. Clona este repositorio:

    git clone https://github.com/adrian-aperez/Product-Rating-Predictor.git

2. Navega al directorio del proyecto:

   cd Product-Rating-Predictor
  
3. Instala las dependencias necesarias:

   pip install -r requirements.txt

4. Puedes ejecutar el código con los datos por defecto en local usando el comando:   

   python -m streamlit run App.py

