import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import numpy as np
import time
from datetime import datetime, timedelta
import pytz

# --- IMPORTACIÓN SEGURA TENSORFLOW ---
try:
    import tensorflow as pd_tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    HAS_TF = True
except ImportError:
    HAS_TF = False

# --- IMPORTACIÓN SEGURA CCXT ---
try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Speedster AI v23", layout="wide", page_icon="⚡")
st.title("⚡ Speedster AI v23 (Memoria Persistente + Fast Signal)")

tz_bolivia = pytz.timezone('America/La_Paz')

# --- ESTADO DE MEMORIA (EL CEREBRO GUARDADO) ---
if 'modelo_ia' not in st.session_state: st.session_state.modelo_ia = None
if 'scaler_ia' not in st.session_state: st.session_state.scaler_ia = None
if 'ultimo_entrenamiento' not in st.session_state: st.session_state.ultimo_entrenamiento = None

# --- SIDEBAR ---
st.sidebar.header("🎛️ Centro de Control")

activos_config = {
    "Bitcoin (BTC)": {"tipo": "crypto", "ticker_ccxt": "BTC/USD", "ticker_y": "BTC-USD"},
    "Solana (SOL)":  {"tipo": "crypto", "ticker_ccxt": "SOL/USD", "ticker_y": "SOL-USD"},
    "EUR/USD":       {"tipo": "forex",  "ticker_ccxt": None,      "ticker_y": "EURUSD=X"},
    "GBP/USD":       {"tipo": "forex",  "ticker_ccxt": None,      "ticker_y": "GBPUSD=X"},
    "USD/JPY":       {"tipo": "forex",  "ticker_ccxt": None,      "ticker_y": "JPY=X"},
}

if 'activo_speed' not in st.session_state: st.session_state.activo_speed = "Solana (SOL)"
def actualizar(): 
    st.session_state.activo_speed = st.session_state.sel_speed
    # Si cambiamos de activo, borramos la memoria para obligar a re-entrenar
    st.session_state.modelo_ia = None 

seleccion = st.sidebar.selectbox("Activo:", list(activos_config.keys()), 
                                 index=list(activos_config.keys()).index(st.session_state.activo_speed), 
                                 key="sel_speed", on_change=actualizar)
datos_activo = activos_config[seleccion]

intervalo = st.sidebar.select_slider("Temporalidad:", options=["1m", "5m", "15m", "1h"], value="1h")
umbral = st.sidebar.slider("Confianza Mínima (%)", 50, 90, 65) # Más bajo para más señales
lookback = 30 # Fijo para optimizar velocidad
vigilancia = st.sidebar.checkbox("🚨 Auto-Escaneo Rápido", value=False)

if not HAS_TF: st.error("Falta TensorFlow. Instálalo para continuar."); st.stop()

# --- 1. MOTOR DE DATOS (OPTIMIZADO) ---
def obtener_datos(config, interval, limite_velas=100):
    # CCXT
    if config['tipo'] == 'crypto' and HAS_CCXT:
        try:
            exchange = ccxt.kraken()
            ohlcv = exchange.fetch_ohlcv(config['ticker_ccxt'], interval, limit=limite_velas)
            df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df.set_index('Datetime', inplace=True)
            df.index = df.index.tz_localize('UTC').tz_convert(None)
            return df, "CCXT"
        except: pass

    # YAHOO
    try:
        # Mapeo de cantidad de datos según si es Training (Mucho) o Live (Poco)
        if limite_velas > 500: # Modo Entrenamiento
            per = "7d" if interval == "1m" else "1y"
        else: # Modo Live (Rápido)
            per = "1d" if interval == "1m" else "5d"
            
        df = yf.download(config['ticker_y'], period=per, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[df['High'] != df['Low']]
        return df, "Yahoo"
    except: return pd.DataFrame(), "Error"

# --- 2. PROCESAMIENTO (AHORA CON ESTOCÁSTICO) ---
def procesar_datos(df):
    if len(df) < 50: return pd.DataFrame()
    
    # Features
    df['RSI'] = df.ta.rsi(length=14)
    
    # Stoch RSI (Más rápido y sensible = Más señales)
    stoch = df.ta.stochrsi(length=14)
    if stoch is not None:
        df['Stoch_K'] = stoch.iloc[:, 0]
        df['Stoch_D'] = stoch.iloc[:, 1]
    
    df['EMA_Fast'] = df.ta.ema(length=9)
    df['ATR'] = df.ta.atr(length=14)
    
    # Target
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# --- 3. CONSTRUCTOR DE MEMORIA ---
def crear_secuencias(data, target, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(target[i])
    return np.array(X), np.array(y)

# --- 4. ENTRENAMIENTO (PESADO - SOLO UNA VEZ) ---
def entrenar_cerebro():
    with st.spinner("🧠 Descargando historia y Entrenando Red Neuronal... (Esto se hace solo una vez)"):
        # 1. Bajamos MUCHOS datos (1500 velas)
        df_train, _ = obtener_datos(datos_activo, intervalo, limite_velas=1500)
        
        if df_train.empty: st.error("No hay datos para entrenar"); return
        
        df_proc = procesar_datos(df_train)
        features = ['Close', 'RSI', 'Stoch_K', 'EMA_Fast', 'ATR']
        features = [f for f in features if f in df_proc.columns]
        
        # 2. Normalizamos y Guardamos el Escalador
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_proc[features].values)
        
        X, y = crear_secuencias(scaled_data[:-1], df_proc['Target'].values[:-1], lookback)
        
        # 3. Creamos Modelo LSTM
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(lookback, len(features))))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # 4. Entrenamos (Más epochs porque lo hacemos una sola vez)
        model.fit(X, y, epochs=15, batch_size=32, verbose=0)
        
        # 5. GUARDAMOS EN SESSION STATE (RAM)
        st.session_state.modelo_ia = model
        st.session_state.scaler_ia = scaler
        st.session_state.features_ia = features
        st.session_state.ultimo_entrenamiento = datetime.now().strftime("%H:%M:%S")
        
        st.success("¡Entrenamiento Completado! Cerebro guardado en memoria.")
        time.sleep(1)
        st.rerun()

# --- 5. PREDICCIÓN (LIGERA - SE EJECUTA SIEMPRE) ---
def predecir_rapido():
    # Bajamos POCOS datos (solo lo necesario para la ventana de memoria)
    # Necesitamos 'lookback' + un extra para calcular indicadores
    df_live, fuente = obtener_datos(datos_activo, intervalo, limite_velas=100)
    
    if df_live.empty: return None, 0.0, pd.DataFrame(), fuente
    
    df_proc = procesar_datos(df_live)
    if df_proc.empty: return None, 0.0, pd.DataFrame(), fuente
    
    # Recuperamos herramientas de la memoria
    model = st.session_state.modelo_ia
    scaler = st.session_state.scaler_ia
    features = st.session_state.features_ia
    
    # Preparamos solo la última secuencia
    last_data = df_proc[features].iloc[-lookback:].values
    last_scaled = scaler.transform(last_data) # Usamos el mismo scaler del entrenamiento
    
    X_pred = last_scaled.reshape(1, lookback, len(features))
    
    prediction = model.predict(X_pred, verbose=0)[0][0]
    
    clase = 1 if prediction > 0.5 else 0
    conf = prediction if clase == 1 else (1 - prediction)
    
    return clase, conf, df_proc, fuente

# --- INTERFAZ ---

# Botón de Reset
col_btn, col_info = st.columns([1, 3])
if col_btn.button("🔄 Re-Entrenar IA"):
    st.session_state.modelo_ia = None
    st.rerun()

if st.session_state.ultimo_entrenamiento:
    col_info.info(f"Cerebro activo desde: {st.session_state.ultimo_entrenamiento}. Modo: Inferencia Rápida ⚡")
else:
    col_info.warning("Cerebro no inicializado.")

placeholder = st.empty()

# LÓGICA PRINCIPAL
with placeholder.container():
    # Si no hay modelo, entrenamos primero
    if st.session_state.modelo_ia is None:
        entrenar_cerebro()
    
    else:
        # Si ya hay modelo, solo predecimos (Rápido)
        pred, conf, df, fuente = predecir_rapido()
        
        if df is not None:
            precio = df['Close'].iloc[-1]
            stoch_k = df['Stoch_K'].iloc[-1]
            
            # Decisión (Más sensible para dar más señales)
            decision = "ESPERAR"
            color = "#e9ecef"; txt = "#333"; icono = "⏳"
            
            if conf*100 >= umbral:
                if pred == 1:
                    decision = "CALL (ALCISTA) 🚀"
                    color = "#d4edda"; txt = "#155724"; icono = "📈"
                    # Señal extra si Stoch está en zona de compra
                    if stoch_k < 20: decision += " [ENTRADA PERFECTA]"
                else:
                    decision = "PUT (BAJISTA) 📉"
                    color = "#f8d7da"; txt = "#721c24"; icono = "📉"
                    if stoch_k > 80: decision += " [ENTRADA PERFECTA]"
            
            # Tiempos
            now = datetime.now(tz_bolivia)
            if intervalo == "1m": seg_rest = 60 - now.second
            elif intervalo == "5m": seg_rest = (5 - (now.minute % 5)) * 60 - now.second
            else: seg_rest = (60 - now.minute) * 60 - now.second
            
            # UI
            c1, c2 = st.columns([2,1])
            c1.markdown(f"### {seleccion} [{intervalo}]")
            c1.markdown(f"<h1 style='margin:0'>${precio:.5f}</h1>", unsafe_allow_html=True)
            
            c2.metric("Vela cierra en:", f"{seg_rest}s")
            c2.caption(f"Fuente: {fuente} | Velocidad: Turbo")
            
            st.markdown(f"""
            <div style="background-color: {color}; padding: 15px; border-radius: 10px; border: 2px solid {txt}; text-align: center;">
                <h1 style="color: {txt}; margin:0;">{icono} {decision}</h1>
                <h2 style="color: {txt}; margin:0;">Confianza: {conf:.1%}</h2>
                <p style="color: {txt}; font-size: 12px;">Stoch RSI: {stoch_k:.1f} (Gatillo Rápido)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Gráfico
            fig = go.Figure(data=[go.Candlestick(x=df.tail(40).index, open=df.tail(40)['Open'], high=df.tail(40)['High'], low=df.tail(40)['Low'], close=df.tail(40)['Close'])])
            fig.update_layout(height=350, template="plotly_white", margin=dict(t=10,b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Error obteniendo datos live.")

if vigilancia:
    time.sleep(3 if intervalo == "1m" else 10) # Refresco mucho más rápido
    st.rerun()
    
