import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Financial Sentiment AI", page_icon="💰")

st.title("💰 Financial Sentiment Analysis")
st.markdown("Clasificación de noticias financieras usando **FinBERT Fine-tuned**.")

# Entrada de texto
text_input = st.text_area(
    "Introduce una frase o noticia financiera en inglés:",
    "The company reported a significant increase in revenue.",
)

if st.button("Analizar Sentimiento"):
    if text_input:
        # Llamada a tu API (asegúrate de que el notebook 06 esté corriendo)
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict", json={"text": text_input}
            )

            if response.status_code == 200:
                data = response.json()
                label = data["label"]
                score = data["score"]

                # Mostrar resultados con colores
                if label == "positive":
                    st.success(
                        f"Sentimiento: **{label.upper()}** (Confianza: {score:.2%})"
                    )
                elif label == "negative":
                    st.error(
                        f"Sentimiento: **{label.upper()}** (Confianza: {score:.2%})"
                    )
                else:
                    st.warning(
                        f"Sentimiento: **{label.upper()}** (Confianza: {score:.2%})"
                    )

                # Gráfico de barra simple
                df_viz = pd.DataFrame({"Métrica": ["Confianza"], "Valor": [score]})
                st.bar_chart(df_viz.set_index("Métrica"))
            else:
                st.error("Error en la API. ¿Está el servidor corriendo?")
        except Exception as e:
            st.error(f"No se pudo conectar con la API: {e}")
    else:
        st.info("Por favor, escribe algo.")
