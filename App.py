import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Productos",
    layout="wide",
    page_icon="🌟"
)

# Título y descripción
st.title(" Predicción de Rating de Productos ")
st.markdown("""
Bienvenido a esta aplicación interactiva para predecir el rating de productos basado en sus características. 
Utiliza un modelo entrenado con datos de productos y sus reseñas. ¡Introduce tus datos y obtén una predicción!
""")

# Cargar modelos y preprocesador
@st.cache_resource
def load_assets():
    preprocessor = joblib.load("preprocessor.pkl")
    model_bajo = joblib.load("modelo_bajo.pkl")
    model_alto = joblib.load("modelo_alto.pkl")
    return preprocessor, model_bajo, model_alto

preprocessor, model_bajo, model_alto = load_assets()

def predict_rating(input_data, preprocessor, model_bajo, model_alto):
    try:
        # Crear un DataFrame con los datos de entrada
        df_input = pd.DataFrame([input_data])
        X_input = df_input[['Tipo', 'Product_Description', 'Price']]

        # Preprocesar los datos
        X_processed = preprocessor.transform(X_input)
        X_processed_dense = X_processed.toarray()

        # Elegir el modelo correcto basado en el número de reseñas
        model = model_bajo if input_data['Reviews'] < 60 else model_alto

        # Realizar la predicción
        prediction = model.predict(X_processed_dense)

        return prediction[0], X_processed_dense
    except Exception as e:
        raise ValueError(f"Error en el preprocesamiento: {e}")

# -------------------------
# Creación de pestañas
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌟 Predicción", 
    "📊 Exploración", 
    "🔍 SHAP", 
    "📈 Informe del producto", 
    "📝 Historial"
])

# -------------------------
# TAB 1: Predicción
# -------------------------
with tab1:
    st.header("🌟 Predicción del Rating")
    with st.form("prediction_form"):
        st.subheader("🧼 Ingrese los datos del producto:")
        tipo = st.selectbox("Tipo de producto", ["champu", "jabon", "exfoliante"])
        # Listas de palabras e ingredientes
        palabras_dominio = [
            'acne', 'aceites', 'afeitado', 'anticaida', 'aroma', 'barba', 'caida', 'canas', 'cara', 'coloracion', 'corporal', 'crecimiento', 'cuerpo',
            'delicada', 'ecologico', 'esencial', 'exfoliante', 'facial', 'fortalecer', 'fragancia', 'graso', 'hidrata', 'marina', 'natural', 'organico',
            'parabenos', 'parfum', 'pelo', 'poros', 'tinte', 'tradicionales', 'vegano'
        ]
        ingredientes_dominio = [
            'aceite de almendras', 'aceite de argan', 'aceite de coco', 'aceite de jojoba', 'aceite de ricino', 'aceite de romero', 'acido-salicilico',
            'aloe vera', 'antioxidantes', 'arcilla', 'cafe', 'cafeina', 'canela', 'carbon', 'citric', 'curcuma', 'glicerina', 'ginseng', 'jengibre', 'madera',
            'mango', 'manteca de karite', 'menta', 'minerales', 'sal rosa', 'sandalo', 'vitamina c', 'zinc'
        ]
        # Bloque 1: Palabras del dominio
        st.subheader("📝 Palabras del Dominio")
        palabras_seleccionadas = st.multiselect("Selecciona palabras:", palabras_dominio, default=[])
        # Bloque 2: Ingredientes
        st.subheader("🌿 Ingredientes")
        ingredientes_seleccionados = st.multiselect("Selecciona ingredientes:", ingredientes_dominio, default=[])
        # Combinar las selecciones en una descripción
        product_description = " ".join(palabras_seleccionadas + ingredientes_seleccionados)
        st.subheader("💶 Precio del producto (€)")
        price = st.number_input("Introduzca el valor", min_value=0.0, step=0.1)
        review_option = st.selectbox("Selecciona una opción", ["Menos de 60 reseñas", "Más de 60 reseñas"])
        # Asignar un valor predeterminado a Reviews
        reviews = 25 if review_option == "Menos de 60 reseñas" else 90
        submitted = st.form_submit_button("Predecir Rating")

    if submitted:
        try:
            # Datos de entrada para la predicción
            input_data = {
                "Tipo": tipo,
                "Product_Description": product_description,
                "Price": price,
                "Reviews": reviews
            }

            # Realizar la predicción
            predicted_rating, X_processed_dense = predict_rating(input_data, preprocessor, model_bajo, model_alto)

            # Guardar todos los valores en session_state
            st.session_state.predicted_rating = float(predicted_rating)
            st.session_state.product_description = product_description
            st.session_state.price = price
            st.session_state.reviews = reviews

            st.success(f"🌟 El rating predicho es: {predicted_rating:.2f} ⭐️")
        except Exception as e:
            st.error(f"❌ Error al hacer la predicción: {str(e)}")
            st.session_state.predicted_rating = None  # Limpiar en caso de error
# -------------------------
# TAB 2: Exploración (EDA)
# -------------------------
with tab2:
    st.header("📊 Análisis Exploratorio de Datos (EDA)")
    st.markdown("Explora las características clave del dataset utilizado para entrenar el modelo.")

    # Cargar datos
    df = pd.read_csv("datos_productos.csv")

    # Resumen de datos
    if st.checkbox("📋 Mostrar resumen de datos"):
        st.write(df.describe())

    # Gráficos interactivos
    st.subheader("📈 Distribución de Ratings")
    fig = px.histogram(
    df,
    x="Star_Rating",
    nbins=10,
    title="Distribución de Ratings",
    color_discrete_sequence=["green"]  # Color de las barras en verde
    )

    # Agregar bordes negros a las barras
    fig.update_traces(marker=dict(line=dict(color="black", width=1)))

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📝 Palabras más frecuentes en Product_Description")
    text = " ".join(df['Product_Description'].dropna())
    wordcloud = WordCloud(width=600, height=290, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Gráfico de dispersión
    st.subheader("🔍 Relación entre Precio y Rating")
    scatter_fig = px.scatter(
        df,
        x="Price",
        y="Star_Rating",
        color="Tipo",
        title="Relación entre Precio y Rating",
        template="plotly_dark"
    )
    st.plotly_chart(scatter_fig, use_container_width=True)


    # Gráfico de dispersión con umbrales
    st.subheader("🔍 Relación entre Reviews y Star_Rating con umbrales")

    # Crear el gráfico de dispersión con Plotly
    scatter_fig = px.scatter(
        df,
        x="Reviews",
        y="Star_Rating",
        color="Tipo",
        title="Relación entre Reviews y Star_Rating",
        template="plotly_dark",  # Tema oscuro
        opacity=0.7,  # Transparencia de los puntos
        labels={"Reviews": "Número de Reviews", "Star_Rating": "Rating"},  # Etiquetas personalizadas
        
    )

    # Agregar líneas de umbral
    scatter_fig.add_hline(
        y=3.5,
        line_dash="dash",
        line_color="blue",
        annotation_text="Umbral Star_Rating (y = 3.5)",
        annotation_position="top right",
    )
    scatter_fig.add_vline(
        x=60,
        line_dash="dash",
        line_color="red",
        annotation_text="Umbral Reviews (x = 60)",
        annotation_position="top left",
    )

    # Personalizar la leyenda
    scatter_fig.update_layout(
        legend=dict(
            orientation="v",  # Orientación vertical
            yanchor="top",  # Alineación vertical
            y=0.5,          # Posición vertical (centrado)
            xanchor="left",  # Alineación horizontal
            x=1.02,         # Posición horizontal (al lado derecho)
            title_text="Tipo"  # Título de la leyenda
        ),
        height=600,  # Altura del gráfico
        margin=dict(l=10, r=10, t=40, b=10),  # Márgenes del gráfico
    )

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(scatter_fig, use_container_width=True)
# -------------------------
# TAB 3: Explicabilidad SHAP
# -------------------------
with tab3:
    st.header("🔍 Explicabilidad del Modelo")
    try:
        # Usar los datos procesados de la predicción anterior
        if 'X_processed_dense' in locals():
            model = model_bajo if reviews < 60 else model_alto
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_processed_dense)

            st.subheader("🧐 Explicación de la Predicción")
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_processed_dense[0]), max_display=10, show=False)
            st.pyplot(fig)
        else:
            st.info("Realiza una predicción en la pestaña 'Predicción' para generar la explicación SHAP.")
    except Exception as e:
        st.error(f"⚠️ No se pudo generar la explicación SHAP: {e}")

# -------------------------
# TAB 4: Informe del Producto
# -------------------------
with tab4:
    st.header("📈 Informe del Producto")
    try:
        # Usar los datos procesados de la predicción anterior
        if 'predicted_rating' in locals():
            # Mostrar métricas principales
            # Personalizar tamaños de fuente
            st.markdown(f"<h2 style='font-size: 21px;'>🌟 Rating Predicho: {predicted_rating:.2f}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 21px;'>🧼 Tipo de producto: {tipo}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 21px;'>📝 Descripción: {product_description}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 21px;'>💵 Precio: €{price:.2f}</p>", unsafe_allow_html=True)

            
            # Mostrar el número de reseñas con el formato deseado
           
            # Mostrar el número de reseñas
            if reviews < 60:
                st.markdown(f"<p style='font-size: 21px;'>📊 Número de Reseñas: Menos de 60</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='font-size: 21px;'>📊 Número de Reseñas: Más de 60</p>", unsafe_allow_html=True)
        else:
            st.info("Realiza una predicción en la pestaña 'Predicción' para generar el informe del producto.")
    except Exception as e:
        st.error(f"⚠️ No se pudo generar el informe del producto: {e}")

# TAB 5: Historial
# -------------------------
with tab5:
    st.header("🕘 Historial de Predicciones")
    
    # Inicializar el historial si no existe
    if "historial" not in st.session_state:
        st.session_state.historial = []
    
    # Guardar caso en el historial - CORRECCIÓN PRINCIPAL
    try:
        # Verificar si hay una predicción válida disponible
        if 'predicted_rating' in st.session_state and st.session_state.get('predicted_rating') is not None:
            if st.button("💾 Guardar este caso"):
                try:
                    # Crear un registro con los datos actuales
                    reviews = st.session_state.get('reviews', 0)
                    registro = {
                        "Rating": round(float(st.session_state.predicted_rating), 2),
                        "Descripción": st.session_state.get('product_description', 'N/A'),
                        "Precio": st.session_state.get('price', 0),
                        "Reseñas": "Más de 60" if reviews >= 60 else "Menos de 60",
                        "Fecha y hora": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    # Guardar el registro en el historial
                    st.session_state.historial.append(registro)
                    st.success("✅ Caso guardado en el historial")
                except Exception as e:
                    st.error(f"⚠️ Error al guardar el caso: {str(e)}")
        else:
            st.info("Realiza una predicción primero en la pestaña 'Predicción' para habilitar esta opción.")
    except Exception as e:
        st.error(f"⚠️ Error inesperado: {str(e)}")
    
    # Mostrar el historial
    if st.session_state.historial:
        st.subheader("📋 Historial de casos guardados")
        df_historial = pd.DataFrame(st.session_state.historial)
        st.dataframe(df_historial)
        
        # Usar columnas para alinear los controles
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Expander para exportar resultados
            with st.expander("Exportar resultados", expanded=False):
                # Contenedor para los botones de exportación
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    # Botón para generar y descargar PDF
                    if st.button("🖨 Generar PDF"):
                        try:
                            def generate_pdf(df):
                                buffer = io.BytesIO()
                                c = canvas.Canvas(buffer, pagesize=letter)
                                
                                # Configuración inicial
                                c.setFont("Helvetica-Bold", 18)
                                c.drawCentredString(300, 750, "Historial de Predicciones")
                                c.setFont("Helvetica", 10)
                                
                                # Encabezado de la tabla
                                x_start = 30
                                y_start = 700
                                column_widths = [70, 200, 60, 70, 100]  # Ajusté los anchos para mejor visualización
                                
                                columns = ["Rating", "Descripción", "Precio", "Reseñas", "Fecha y hora"]
                                
                                # Dibujar encabezados
                                x_pos = x_start
                                for i, col in enumerate(columns):
                                    c.setFont("Helvetica-Bold", 10)
                                    c.drawString(x_pos, y_start, col)
                                    x_pos += column_widths[i]
                                
                                # Línea bajo encabezados
                                c.line(x_start, y_start-10, x_start + sum(column_widths), y_start-10)
                                
                                # Datos de la tabla
                                y_pos = y_start - 25
                                for _, row in df.iterrows():
                                    x_pos = x_start
                                    for i, value in enumerate(row):
                                        c.setFont("Helvetica", 9)  # Reduje el tamaño de fuente
                                        text = str(value)
                                        
                                        # Manejar texto largo para descripción
                                        if i == 1 and len(text) > 25:  # Ajusté el límite de caracteres
                                            parts = [text[j:j+25] for j in range(0, len(text), 25)]
                                            for k, part in enumerate(parts):
                                                c.drawString(x_pos, y_pos - (k*12), part)
                                        else:
                                            c.drawString(x_pos, y_pos, text)
                                        
                                        x_pos += column_widths[i]
                                    
                                    # Línea horizontal entre filas
                                    c.line(x_start, y_pos-15, x_start + sum(column_widths), y_pos-15)
                                    y_pos -= 30 + (12 * ((len(text)//25) if i == 1 and len(text) > 25 else 0))
                                    
                                    # Nueva página si es necesario
                                    if y_pos < 50:
                                        c.showPage()
                                        y_pos = 700
                                        c.setFont("Helvetica-Bold", 18)
                                        c.drawCentredString(300, 750, "Historial de Predicciones (cont.)")
                                        c.setFont("Helvetica", 10)
                                        x_pos = x_start
                                        for i, col in enumerate(columns):
                                            c.setFont("Helvetica-Bold", 10)
                                            c.drawString(x_pos, y_pos, col)
                                            x_pos += column_widths[i]
                                        c.line(x_start, y_pos-10, x_start + sum(column_widths), y_pos-10)
                                        y_pos -= 25
                                
                                c.save()
                                buffer.seek(0)
                                return buffer
                            
                            pdf_buffer = generate_pdf(df_historial)
                            st.download_button(
                                label="⬇️ Descargar PDF",
                                data=pdf_buffer.getvalue(),
                                file_name='historial_predicciones.pdf',
                                mime='application/pdf',
                                key='pdf_download'
                            )
                        except Exception as e:
                            st.error(f"⚠️ Error al generar el PDF: {str(e)}")
                
                with export_col2:
                    # Botón para descargar CSV
                    csv_data = df_historial.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Descargar CSV",
                        data=csv_data,
                        file_name='historial_predicciones.csv',
                        mime='text/csv',
                        key='csv_download'
                    )
        
        with col2:
            # Botón para limpiar el historial
            if st.button("🧹 Limpiar historial", key="clear_history"):
                st.session_state.historial = []
                st.success("Historial limpiado correctamente")
    else:
        st.info("No hay casos guardados aún.")
# Footer
st.markdown("---")
st.markdown("Made by Adrián Alonso Pérez")


st.markdown("[GitHub Repository](https://github.com/tu-usuario/tu-repositorio)")
st.markdown("[LinkedIn Profile](https://www.linkedin.com/in/tu-perfil/)")

