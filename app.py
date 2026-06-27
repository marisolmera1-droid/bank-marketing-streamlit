
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# =========================
# CONFIGURACIÓN
# =========================

st.set_page_config(
    page_title="Clasificador Bank Marketing",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.titulo {
    font-size: 42px;
    font-weight: bold;
    color: #1f3b73;
}
.subtitulo {
    font-size: 20px;
    color: #444;
}
.caja {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# =========================
# CARGA DE ARCHIVOS
# =========================

modelo = joblib.load("modelo_arbol.pkl")
columnas_modelo = joblib.load("columnas_modelo.pkl")
df = pd.read_csv("banking.csv")

# =========================
# PORTADA
# =========================

st.markdown('<p class="titulo">Clasificador de Clientes - Marketing Bancario</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitulo">Aplicación interactiva para visualizar resultados del modelo y probar predicciones de clientes.</p>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs([
    "📊 Fase 1: Resultados del clasificador",
    "📈 Visualizaciones interactivas",
    "🧑‍💼 Fase 2: Prueba del modelo"
])

# =========================
# FASE 1 RESULTADOS
# =========================

with tab1:
    st.header("Fase 1: Resultados del clasificador")

    st.write("""
    En esta sección se presentan los resultados obtenidos durante la evaluación de los modelos de clasificación.
    El modelo seleccionado fue el **Árbol de Decisión**, debido a que presentó el mejor equilibrio entre Accuracy,
    Precision, Recall y F1-Score para la clase positiva.
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "0.91")
    col2.metric("Precision clase 1", "0.65")
    col3.metric("Recall clase 1", "0.52")
    col4.metric("F1-Score clase 1", "0.58")

    resultados = pd.DataFrame({
        "Modelo": ["Naive Bayes", "Árbol de Decisión", "Random Forest", "MLP"],
        "Accuracy": [0.75, 0.91, 0.91, 0.90],
        "Precision clase 1": [0.28, 0.65, 0.65, 0.55],
        "Recall clase 1": [0.79, 0.52, 0.47, 0.54],
        "F1-Score clase 1": [0.41, 0.58, 0.55, 0.54]

    })

    st.subheader("Comparación de modelos")
    html_tabla = """
    <table>
    <tr>
    <th>Modelo</th><th>Accuracy</th><th>Precision clase 1</th><th>Recall clase 1</th><th>F1-Score clase 1</th>
    </tr>
    <tr><td>Naive Bayes</td><td>0,75</td><td>0,28</td><td>0,79</td><td>0,41</td></tr>
    <tr><td>Árbol de Decisión</td><td>0,91</td><td>0,65</td><td>0,52</td><td>0,58</td></tr>
    <tr><td>Random Forest</td><td>0,91</td><td>0,65</td><td>0,47</td><td>0,55</td></tr>
    <tr><td>MLP</td><td>0,90</td><td>0,55</td><td>0,54</td><td>0,54</td></tr>
    </table>
"""
    st.markdown(html_tabla, unsafe_allow_html=True)

    fig = px.bar(
        resultados,
        x="Modelo",
        y=["Accuracy", "Precision clase 1", "Recall clase 1", "F1-Score clase 1"],
        barmode="group",
        title="Comparación de métricas por modelo"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    Interpretación: El Árbol de Decisión fue seleccionado porque mantiene una exactitud alta y un mejor equilibrio
    en la detección de clientes que podrían aceptar el depósito a plazo.
    """)

# =========================
# VISUALIZACIONES
# =========================

with tab2:
    st.header("Visualizaciones interactivas del dataset")

    if "y" in df.columns:
        conteo_y = df["y"].value_counts().reset_index()
        conteo_y.columns = ["Respuesta", "Cantidad"]

        fig_y = px.pie(
            conteo_y,
            names="Respuesta",
            values="Cantidad",
            title="Distribución de clientes según aceptación del depósito"
        )
        st.plotly_chart(fig_y, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        if "age" in df.columns:
            fig_age = px.histogram(
                df,
                x="age",
                color="y" if "y" in df.columns else None,
                nbins=30,
                title="Distribución de edad de los clientes"
            )
            st.plotly_chart(fig_age, use_container_width=True)

    with col_b:
        if "duration" in df.columns:
            fig_duration = px.box(
                df,
                x="y" if "y" in df.columns else None,
                y="duration",
                title="Duración del contacto según resultado"
            )
            st.plotly_chart(fig_duration, use_container_width=True)

    if "job" in df.columns and "y" in df.columns:
        job_data = df.groupby(["job", "y"]).size().reset_index(name="Cantidad")

        fig_job = px.bar(
            job_data,
            x="job",
            y="Cantidad",
            color="y",
            title="Resultado de campaña según ocupación",
            barmode="group"
        )
        st.plotly_chart(fig_job, use_container_width=True)

# =========================
# FASE 2 PRUEBA DEL MODELO
# =========================

with tab3:
    st.header("Fase 2: Prueba interactiva del modelo")

    st.write("""
    En esta fase el usuario puede ingresar manualmente las características de un cliente.
    Luego, el modelo predice si el cliente probablemente aceptará o no el depósito a plazo.
    """)

    st.subheader("Ingrese los datos del cliente")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Edad", min_value=18, max_value=100, value=35)
        duration = st.number_input("Duración del contacto", min_value=0, value=120)
        campaign = st.number_input("Número de contactos campaña", min_value=1, value=1)
        pdays = st.number_input("Días desde último contacto", min_value=-1, value=999)
        previous = st.number_input("Contactos previos", min_value=0, value=0)

    with col2:
        job = st.selectbox("Ocupación", sorted(df["job"].dropna().unique()) if "job" in df.columns else ["admin."])
        marital = st.selectbox("Estado civil", sorted(df["marital"].dropna().unique()) if "marital" in df.columns else ["married"])
        education = st.selectbox("Educación", sorted(df["education"].dropna().unique()) if "education" in df.columns else ["university.degree"])
        default = st.selectbox("Crédito en default", sorted(df["default"].dropna().unique()) if "default" in df.columns else ["no"])
        housing = st.selectbox("Crédito hipotecario", sorted(df["housing"].dropna().unique()) if "housing" in df.columns else ["no"])

    with col3:
        loan = st.selectbox("Préstamo personal", sorted(df["loan"].dropna().unique()) if "loan" in df.columns else ["no"])
        contact = st.selectbox("Tipo de contacto", sorted(df["contact"].dropna().unique()) if "contact" in df.columns else ["cellular"])
        month = st.selectbox("Mes", sorted(df["month"].dropna().unique()) if "month" in df.columns else ["may"])
        day_of_week = st.selectbox("Día de la semana", sorted(df["day_of_week"].dropna().unique()) if "day_of_week" in df.columns else ["mon"])
        poutcome = st.selectbox("Resultado campaña anterior", sorted(df["poutcome"].dropna().unique()) if "poutcome" in df.columns else ["nonexistent"])

    emp_var_rate = st.number_input("Emp.var.rate", value=float(df["emp.var.rate"].median()) if "emp.var.rate" in df.columns else 1.1)
    cons_price_idx = st.number_input("Cons.price.idx", value=float(df["cons.price.idx"].median()) if "cons.price.idx" in df.columns else 93.2)
    cons_conf_idx = st.number_input("Cons.conf.idx", value=float(df["cons.conf.idx"].median()) if "cons.conf.idx" in df.columns else -40.0)
    euribor3m = st.number_input("Euribor 3 meses", value=float(df["euribor3m"].median()) if "euribor3m" in df.columns else 4.8)
    nr_employed = st.number_input("N° empleados", value=float(df["nr.employed"].median()) if "nr.employed" in df.columns else 5191.0)

    st.caption("Clase 1 = cliente acepta el depósito. Clase 0 = cliente no acepta.")

    if st.button("Predecir resultado del cliente"):

        nuevo_cliente = pd.DataFrame({
            "age": [age],
            "job": [job],
            "marital": [marital],
            "education": [education],
            "default": [default],
            "housing": [housing],
            "loan": [loan],
            "contact": [contact],
            "month": [month],
            "day_of_week": [day_of_week],
            "duration": [duration],
            "campaign": [campaign],
            "pdays": [pdays],
            "previous": [previous],
            "poutcome": [poutcome],
            "emp.var.rate": [emp_var_rate],
            "cons.price.idx": [cons_price_idx],
            "cons.conf.idx": [cons_conf_idx],
            "euribor3m": [euribor3m],
            "nr.employed": [nr_employed]
        })

        nuevo_cliente_codificado = pd.get_dummies(nuevo_cliente)
        nuevo_cliente_codificado = nuevo_cliente_codificado.reindex(columns=columnas_modelo, fill_value=0)

        prediccion = modelo.predict(nuevo_cliente_codificado)

        st.subheader("Resultado de la predicción")

        if prediccion[0] == 1:
            st.success("El cliente probablemente ACEPTA el depósito a plazo.")
        else:
            st.warning("El cliente probablemente NO acepta el depósito a plazo.")

        st.write("Datos ingresados:")
        st.markdown(nuevo_cliente.to_html(index=False), unsafe_allow_html=True)
