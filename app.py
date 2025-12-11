import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from datetime import timedelta

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Forex AI Predictor", layout="wide")

st.title("💶 Predicción EUR/USD con Inteligencia Artificial")
st.markdown("""
Esta aplicación utiliza **Random Forest** (Machine Learning) para analizar 
patrones técnicos históricos y predecir la dirección del mercado.
""")

# --- BARRA LATERAL (CONTROLES) ---
st.sidebar.header("Configuración del Análisis")
periodo = st.sidebar.selectbox("Periodo Histórico", ["5y", "10y", "20y", "max"], index=2)
dias_entrenamiento = st.sidebar.slider("Días para medias móviles (Corto Plazo)", 10, 50, 50)

# --- FUNCIÓN DE CARGA DE DATOS (CORREGIDA) ---
@st.cache_data
def cargar_datos(periodo_ticker):
    # Descargamos datos diarios
    df = yf.download("EURUSD=X", period=periodo_ticker)
    
    # === AQUÍ ESTÁ EL ARREGLO ===
    # Si yfinance nos devuelve columnas dobles (MultiIndex), las aplanamos
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # ============================
    
    # Aseguramos que el índice es datetime y eliminamos zona horaria
    df.index = pd.to_datetime(df.index).tz_localize(None) 
    
    # 1. INDICADORES TÉCNICOS
    df['RSI'] = df.ta.rsi(length=14)
    df['EMA_Fast'] = df.ta.ema(length=dias_entrenamiento)
    df['EMA_Slow'] = df.ta.ema(length=200)
    
    # Bandas de Bollinger
    bb = df.ta.bbands(length=20)
    df = pd.concat([df, bb], axis=1)

    # 2. DEFINIR EL TARGET
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.dropna(inplace=True)
    return df


# Cargar los datos
status_text = st.sidebar.text("Descargando datos del mercado...")
df = cargar_datos(periodo)
status_text.text("Datos cargados y procesados.")

# --- VISUALIZACIÓN FINANCIERA (PLOTLY) ---
st.subheader("Gráfico de Mercado (Velas Japonesas)")

# Tomamos solo los últimos 150 días para que el gráfico sea legible
df_visual = df.tail(150)

fig = go.Figure(data=[go.Candlestick(x=df_visual.index,
                open=df_visual['Open'],
                high=df_visual['High'],
                low=df_visual['Low'],
                close=df_visual['Close'],
                name="EUR/USD")])

# Añadir medias móviles al gráfico
fig.add_trace(go.Scatter(x=df_visual.index, y=df_visual['EMA_Fast'], line=dict(color='orange', width=1), name=f"EMA {dias_entrenamiento}"))
fig.add_trace(go.Scatter(x=df_visual.index, y=df_visual['EMA_Slow'], line=dict(color='blue', width=1), name="EMA 200"))

fig.update_layout(xaxis_rangeslider_visible=False, height=500)
st.plotly_chart(fig, use_container_width=True)

# --- ENTRENAMIENTO DEL MODELO ---
if st.button('🧠 Entrenar Modelo y Predecir'):
    
    # Variables que usará la IA para decidir
    features = ['RSI', 'EMA_Fast', 'EMA_Slow', 'Open', 'Close', 'High', 'Low']
    
    # Separar datos (Entrenamiento vs Test)
    # Usamos los últimos 500 días para testear la precisión reciente
    train = df.iloc[:-500]
    test = df.iloc[-500:]
    
    # Modelo Random Forest (Robusto contra el ruido)
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=42)
    
    # Entrenar
    model.fit(train[features], train["Target"])
    
    # Evaluar precisión
    preds = model.predict(test[features])
    precision = precision_score(test["Target"], preds)
    
    # --- RESULTADOS ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"Precisión del modelo (Backtest reciente): {precision:.2%}")
        if precision < 0.5:
            st.warning("⚠️ La precisión actual es baja. El mercado está muy volátil.")
            
    with col2:
        # Predicción para mañana usando el ÚLTIMO dato disponible hoy
        ultimo_dia = df.iloc[-1:][features]
        prediccion = model.predict(ultimo_dia)
        
        resultado = "SUBIRÁ 📈" if prediccion[0] == 1 else "BAJARÁ 📉"
        
        st.metric(label="Predicción para el siguiente cierre", value=resultado)
        
    st.write("---")
    st.caption("Nota: Los mercados financieros conllevan riesgo. Esta herramienta es para fines educativos.")
 
