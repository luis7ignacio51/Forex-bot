import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import numpy as np
import time
from datetime import datetime, timedelta
import pytz

# --- INTENTO DE IMPORTAR TENSORFLOW ---
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
st.set_page_config(page_title="LSTM Pro Trader v22", layout="wide", page_icon="🧠")
st.title("🧠 LSTM Pro Trader v22 (Deep Learning + MACD)")

tz_bolivia = pytz.timezone('America/La_Paz')

# --- SIDEBAR ---
st.sidebar.header("🎛️ Centro de Control Neural")

activos_config = {
    "Bitcoin (BTC)": {"tipo": "crypto", "ticker_ccxt": "BTC/USD", "ticker_y": "BTC-USD"},
    "Solana (SOL)":  {"tipo": "crypto", "ticker_ccxt": "SOL/USD", "ticker_y": "SOL-USD"},
    "EUR/USD":       {"tipo": "forex",  "ticker_ccxt": None,      "ticker_y": "EURUSD=X"},
    "GBP/USD":       {"tipo": "forex",  "ticker_ccxt": None,      "ticker_y": "GBPUSD=X"},
    "USD/JPY":       {"tipo": "forex",  "ticker_ccxt": None,      "ticker_y": "JPY=X"},
}

if 'activo_lstm' not in st.session_state: st.session_state.activo_lstm = "Solana (SOL)"
def actualizar(): st.session_state.activo_lstm = st.session_state.sel_lstm
seleccion = st.sidebar.selectbox("Activo:", list(activos_config.keys()), index=list(activos_config.keys()).index(st.session_state.activo_lstm), key="sel_lstm", on_change=actualizar)
datos_activo = activos_config[seleccion]

intervalo = st.sidebar.select_slider("Temporalidad:", options=["1m", "5m", "15m", "1h"], value="1h")
# Bajamos un poco el umbral por defecto porque las LSTM son más conservadoras
umbral = st.sidebar.slider("Confianza Mínima (%)", 55, 95, 70) 
lookback = st.sidebar.slider("Memoria (Velas atrás)", 10, 60, 30)
vigilancia = st.sidebar.checkbox("🚨 Auto-Escaneo", value=False)

if not HAS_TF:
    st.error("❌ TensorFlow no instalado. Agrega 'tensorflow' a requirements.txt")
    st.stop()

# --- 1. MOTOR DE DATOS (BOOSTED) ---
@st.cache_data(ttl=30)
def obtener_datos(config, interval):
    # A. CCXT (Cripto)
    if config['tipo'] == 'crypto' and HAS_CCXT:
        try:
            exchange = ccxt.kraken()
            # AUMENTO DE DATOS: De 500 a 1000 velas para que la IA aprenda más
            limit = 1000 
            ohlcv = exchange.fetch_ohlcv(config['ticker_ccxt'], interval, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df.set_index('Datetime', inplace=True)
            df.index = df.index.tz_localize('UTC').tz_convert(None)
            return df, "CCXT (Kraken)"
        except: pass

    # B. YAHOO
    try:
        # Pedimos más historia para Forex también
        y_per = "7d" if interval == "1m" else ("60d" if interval in ["5m", "15m"] else "1y")
        df = yf.download(config['ticker_y'], period=y_per, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[df['High'] != df['Low']] # Limpieza
        return df, "Yahoo Finance"
    except: return pd.DataFrame(), "Error"

# --- 2. PREPARACIÓN DE DATOS (FEATURES + MACD) ---
def procesar_datos(df):
    if len(df) < 100: return pd.DataFrame()
    
    # Features Clásicos
    df['RSI'] = df.ta.rsi(length=14)
    df['EMA_Fast'] = df.ta.ema(length=9)
    df['EMA_Slow'] = df.ta.ema(length=21)
    df['ATR'] = df.ta.atr(length=14)
    
    # NUEVO: MACD (Ayuda a detectar cambios de tendencia mejor que RSI solo)
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    if macd is not None:
        df['MACD'] = macd.iloc[:, 0] # Línea MACD
        df['MACD_Signal'] = macd.iloc[:, 2] # Histograma
    
    # Target: 1 si Close sube, 0 si baja
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# --- 3. SECUENCIAS ---
def crear_secuencias(data, target, lookback_period):
    X, y = [], []
    for i in range(lookback_period, len(data)):
        X.append(data[i-lookback_period:i]) 
        y.append(target[i])
    return np.array(X), np.array(y)

# --- 4. CEREBRO LSTM ---
def ejecutar_lstm(df, lookback_period):
    # Agregamos MACD a la visión de la IA
    features = ['Close', 'RSI', 'EMA_Fast', 'EMA_Slow', 'ATR', 'MACD', 'MACD_Signal']
    # Validar que existan
    features = [f for f in features if f in df.columns]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features].values)
    target_data = df['Target'].values
    
    X, y = crear_secuencias(scaled_data[:-1], target_data[:-1], lookback_period)
    
    if len(X) < 50: return 0, 0.0
    
    # Modelo
    model = Sequential()
    # Aumentamos neuronas ligeramente
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Aumentamos epochs de 5 a 10 para mejor aprendizaje (sin sacrificar tanta velocidad)
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    
    # Predicción
    last_sequence = scaled_data[-lookback_period:]
    last_sequence = last_sequence.reshape(1, lookback_period, len(features))
    
    prediction = model.predict(last_sequence)[0][0]
    
    clase = 1 if prediction > 0.5 else 0
    confianza = prediction if clase == 1 else (1 - prediction)
    
    return clase, confianza

# --- 5. INTERFAZ ---
if st.sidebar.button("🧠 Re-Entrenar Cerebro"): st.cache_data.clear()
placeholder = st.empty()

with placeholder.container():
    df_raw, fuente = obtener_datos(datos_activo, intervalo)
    
    if not df_raw.empty:
        df_proc = procesar_datos(df_raw)
        
        if not df_proc.empty:
            # Spinner informativo
            with st.spinner(f"Analizando {len(df_proc)} velas secuenciales con MACD..."):
                pred, conf = ejecutar_lstm(df_proc, lookback)
            
            # Datos visuales
            precio = df_proc['Close'].iloc[-1]
            
            # --- CÁLCULO DE TIEMPO (RESTORED) ---
            now = datetime.now(tz_bolivia)
            minutos = now.minute
            
            if intervalo == "1m": 
                segundos_restantes = 60 - now.second
                delta = timedelta(seconds=segundos_restantes)
            elif intervalo == "5m": 
                segundos_restantes = (5 - (minutos % 5)) * 60 - now.second
                delta = timedelta(seconds=segundos_restantes)
            elif intervalo == "15m":
                segundos_restantes = (15 - (minutos % 15)) * 60 - now.second
                delta = timedelta(seconds=segundos_restantes)
            else: # 1h
                segundos_restantes = (60 - minutos) * 60 - now.second
                delta = timedelta(seconds=segundos_restantes)
            
            hora_prediccion = (now + delta).strftime("%H:%M:%S")
            
            # Lógica Decisión
            decision = "NEUTRO / ESPERAR"
            color = "#e9ecef"; txt = "#333"; icono = "⚖️"
            
            if conf*100 >= umbral:
                if pred == 1:
                    decision = "PROYECCIÓN ALCISTA 🚀"
                    color = "#d4edda"; txt = "#155724"; icono = "📈"
                else:
                    decision = "PROYECCIÓN BAJISTA 📉"
                    color = "#f8d7da"; txt = "#721c24"; icono = "📉"
            else:
                # Mensaje explicativo si la confianza es baja
                decision = f"ESPERAR (Confianza IA: {conf:.1%})"
                icono = "✋"

            # UI
            c1, c2 = st.columns([2,1])
            c1.markdown(f"### {seleccion} [{intervalo}]")
            c1.markdown(f"<h1 style='margin:0'>${precio:.5f}</h1>", unsafe_allow_html=True)
            
            # INFO DE LA VELA Y PREDICCIÓN
            c2.metric("Cierre Vela:", f"{segundos_restantes}s")
            c2.markdown(f"**🎯 Predicción para vela de las:** `{hora_prediccion}`")
            c2.caption(f"Fuente: {fuente} | Datos: {len(df_proc)} velas")
            
            st.markdown(f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 15px; border: 3px solid {txt}; text-align: center;">
                <h2 style="color: {txt}; margin:0;">{icono} {decision}</h2>
                <h1 style="color: {txt}; margin:0; font-size: 50px;">{conf:.1%}</h1>
                <p style="color: {txt};">Certeza Neuronal (Basada en {lookback} velas previas)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Gráfico
            fig = go.Figure(data=[go.Candlestick(x=df_proc.tail(60).index, open=df_proc.tail(60)['Open'], high=df_proc.tail(60)['High'], low=df_proc.tail(60)['Low'], close=df_proc.tail(60)['Close'])])
            # Añadimos EMA para referencia visual
            fig.add_trace(go.Scatter(x=df_proc.tail(60).index, y=df_proc.tail(60)['EMA_Slow'], line=dict(color='blue', width=1), name="EMA 21"))
            fig.update_layout(height=350, template="plotly_white", margin=dict(t=10,b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
        else: st.warning("Recopilando datos suficientes para la Red Neuronal...")
    else: st.error("Error de conexión con el proveedor de datos.")

if vigilancia:
    time.sleep(10 if intervalo == "1m" else 30)
    st.rerun()
    
