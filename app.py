import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import numpy as np
import time
from datetime import datetime
import pytz

# --- INTENTO DE IMPORTAR TENSORFLOW (CEREBRO REAL) ---
try:
    import tensorflow as pd_tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    HAS_TF = True
except ImportError:
    HAS_TF = False

# --- IMPORTACIÓN SEGURA DE CCXT ---
try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LSTM Neural Trader", layout="wide", page_icon="🧠")
st.title("🧠 LSTM Neural Trader (Deep Learning)")

tz_bolivia = pytz.timezone('America/La_Paz')

# --- SIDEBAR ---
st.sidebar.header("🎛️ Configuración Neuronal")

activos_config = {
    "Bitcoin (BTC)": {"tipo": "crypto", "ticker_ccxt": "BTC/USD", "ticker_y": "BTC-USD"},
    "Solana (SOL)":  {"tipo": "crypto", "ticker_ccxt": "SOL/USD", "ticker_y": "SOL-USD"},
    "EUR/USD":       {"tipo": "forex",  "ticker_ccxt": None,      "ticker_y": "EURUSD=X"},
    "GBP/USD":       {"tipo": "forex",  "ticker_ccxt": None,      "ticker_y": "GBPUSD=X"},
}

if 'activo_lstm' not in st.session_state: st.session_state.activo_lstm = "Solana (SOL)"
def actualizar(): st.session_state.activo_lstm = st.session_state.sel_lstm
seleccion = st.sidebar.selectbox("Activo:", list(activos_config.keys()), index=list(activos_config.keys()).index(st.session_state.activo_lstm), key="sel_lstm", on_change=actualizar)
datos_activo = activos_config[seleccion]

intervalo = st.sidebar.select_slider("Temporalidad:", options=["1m", "5m", "15m", "1h"], value="1h")
umbral = st.sidebar.slider("Confianza Mínima (%)", 60, 95, 75)
lookback = st.sidebar.slider("Memoria (Velas atrás)", 10, 60, 30) # Cuántas velas mira hacia atrás
vigilancia = st.sidebar.checkbox("🚨 Auto-Escaneo", value=False)

if not HAS_TF:
    st.error("❌ TensorFlow no está instalado. Agrega 'tensorflow' a requirements.txt para usar LSTMs.")
    st.stop()

# --- 1. MOTOR DE DATOS (TURBO) ---
@st.cache_data(ttl=30)
def obtener_datos(config, interval):
    # A. CCXT
    if config['tipo'] == 'crypto' and HAS_CCXT:
        try:
            exchange = ccxt.kraken()
            # Necesitamos MÁS datos para LSTMs (mínimo 300 velas para entrenar bien)
            limit = 500 
            ohlcv = exchange.fetch_ohlcv(config['ticker_ccxt'], interval, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df.set_index('Datetime', inplace=True)
            df.index = df.index.tz_localize('UTC').tz_convert(None)
            return df, "CCXT"
        except: pass

    # B. YAHOO
    try:
        y_per = "5d" if interval == "1m" else "60d" # Pedimos bastantes datos
        df = yf.download(config['ticker_y'], period=y_per, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[df['High'] != df['Low']]
        return df, "Yahoo"
    except: return pd.DataFrame(), "Error"

# --- 2. PREPARACIÓN DE DATOS PARA RED NEURONAL ---
def procesar_datos(df):
    if len(df) < 100: return pd.DataFrame(), None
    
    # Indicadores Técnicos (Features)
    df['RSI'] = df.ta.rsi(length=14)
    df['EMA_Fast'] = df.ta.ema(length=9)
    df['EMA_Slow'] = df.ta.ema(length=21)
    df['ATR'] = df.ta.atr(length=14)
    
    # Target: 1 si Close sube, 0 si baja
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.dropna(inplace=True)
    return df

# --- 3. CONSTRUCCIÓN DE SECUENCIAS LSTM ---
def crear_secuencias(data, target, lookback_period):
    X, y = [], []
    for i in range(lookback_period, len(data)):
        X.append(data[i-lookback_period:i]) # Agarra las X velas anteriores
        y.append(target[i])
    return np.array(X), np.array(y)

# --- 4. CEREBRO LSTM (DEEP LEARNING) ---
def ejecutar_lstm(df, lookback_period):
    # Seleccionamos las columnas que la IA va a "mirar"
    features = ['Close', 'RSI', 'EMA_Fast', 'EMA_Slow', 'ATR']
    
    # NORMALIZACIÓN (CRUCIAL PARA REDES NEURONALES)
    # Las LSTMs funcionan mejor con números entre 0 y 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features].values)
    target_data = df['Target'].values
    
    # Crear secuencias (X=Pasado, y=Futuro)
    # Usamos todo menos la última fila para entrenar
    X, y = crear_secuencias(scaled_data[:-1], target_data[:-1], lookback_period)
    
    if len(X) < 50: return 0, 0.0 # Seguridad
    
    # Construcción del Modelo LSTM
    model = Sequential()
    # Capa 1: LSTM con memoria
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2)) # Apaga 20% neuronas para evitar memorización tonta
    # Capa 2: LSTM final
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    # Capa Salida: 0 a 1 (Probabilidad)
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # ENTRENAMIENTO (Epochs bajito para velocidad en vivo)
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    
    # PREDICCIÓN LIVE
    # Tomamos las últimas 'lookback' velas reales para predecir la siguiente
    last_sequence = scaled_data[-lookback_period:]
    last_sequence = last_sequence.reshape(1, lookback_period, len(features))
    
    prediction = model.predict(last_sequence)[0][0] # Resultado entre 0 y 1
    
    # Decisión binaria
    clase = 1 if prediction > 0.5 else 0
    # La confianza es la distancia al extremo (0 o 1)
    confianza = prediction if clase == 1 else (1 - prediction)
    
    return clase, confianza

# --- 5. INTERFAZ ---
if st.sidebar.button("🧠 Entrenar Red Neuronal"): st.cache_data.clear()
placeholder = st.empty()

with placeholder.container():
    df_raw, fuente = obtener_datos(datos_activo, intervalo)
    
    if not df_raw.empty:
        df_proc = procesar_datos(df_raw)
        
        if not df_proc.empty:
            # Spinner visual mientras la red entrena
            with st.spinner(f"Entrenando LSTM con {len(df_proc)} velas y {lookback} periodos de memoria..."):
                pred, conf = ejecutar_lstm(df_proc, lookback)
            
            # Datos
            precio = df_proc['Close'].iloc[-1]
            rsi = df_proc['RSI'].iloc[-1]
            
            # Lógica Visual
            decision = "NEUTRO"
            color = "#e9ecef"; txt = "#333"
            
            if conf*100 >= umbral:
                if pred == 1:
                    decision = "PROYECCIÓN ALCISTA (CALL) 🚀"
                    color = "#d4edda"; txt = "#155724"
                else:
                    decision = "PROYECCIÓN BAJISTA (PUT) 📉"
                    color = "#f8d7da"; txt = "#721c24"
            else:
                decision = "ESPERAR (Incertidumbre)"
            
            # UI
            c1, c2 = st.columns([2,1])
            c1.markdown(f"### {seleccion} [{intervalo}]")
            c1.markdown(f"<h1 style='margin:0'>${precio:.5f}</h1>", unsafe_allow_html=True)
            c2.caption(f"Fuente: {fuente} | Deep Learning Active")
            
            st.markdown(f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 15px; border: 2px solid {txt}; text-align: center;">
                <h2 style="color: {txt}; margin:0;">{decision}</h2>
                <h1 style="color: {txt}; margin:0; font-size: 40px;">{conf:.1%}</h1>
                <p style="color: {txt};">Probabilidad Calculada por Red Neuronal</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Gráfico
            fig = go.Figure(data=[go.Candlestick(x=df_proc.tail(60).index, open=df_proc.tail(60)['Open'], high=df_proc.tail(60)['High'], low=df_proc.tail(60)['Low'], close=df_proc.tail(60)['Close'])])
            fig.update_layout(height=350, template="plotly_white", margin=dict(t=10,b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("🤓 ¿Cómo piensa esta Red Neuronal?"):
                st.write(f"""
                Esta **LSTM** no mira solo el precio actual. 
                1. Toma las últimas **{lookback} velas**.
                2. Normaliza los precios (0 a 1).
                3. Busca patrones secuenciales en Close, RSI y EMAs.
                4. Aplica "Dropout" para ignorar ruido.
                """)

        else: st.warning("Recopilando datos...")
    else: st.error("Error de conexión.")

if vigilancia:
    time.sleep(10 if intervalo == "1m" else 30)
    st.rerun()
    
