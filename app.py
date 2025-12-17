import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import numpy as np
import time
from datetime import datetime, timedelta
import pytz

# --- IMPORTACIONES SEGURAS ---
try:
    import tensorflow as pd_tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    HAS_TF = True
except ImportError: HAS_TF = False

try:
    import ccxt
    HAS_CCXT = True
except ImportError: HAS_CCXT = False

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Market God AI v25", layout="wide", page_icon="🛡️")
st.title("🛡️ Market God AI v25 (Fixed H1 & Data)")

tz_bolivia = pytz.timezone('America/La_Paz')

# --- ESTADO DE MEMORIA ---
if 'modelo_ia' not in st.session_state: st.session_state.modelo_ia = None
if 'scaler_ia' not in st.session_state: st.session_state.scaler_ia = None

# --- SIDEBAR ---
st.sidebar.header("🎛️ Centro de Comando")

activos_config = {
    "Bitcoin (BTC)": {"tipo": "crypto", "ticker_ccxt": "BTC/USD", "ticker_y": "BTC-USD"},
    "Solana (SOL)":  {"tipo": "crypto", "ticker_ccxt": "SOL/USD", "ticker_y": "SOL-USD"},
    "EUR/USD":       {"tipo": "forex",  "ticker_ccxt": None,      "ticker_y": "EURUSD=X"},
    "GBP/USD":       {"tipo": "forex",  "ticker_ccxt": None,      "ticker_y": "GBPUSD=X"},
    "USD/JPY":       {"tipo": "forex",  "ticker_ccxt": None,      "ticker_y": "JPY=X"},
}

if 'activo_god' not in st.session_state: st.session_state.activo_god = "Solana (SOL)"
def actualizar(): 
    st.session_state.activo_god = st.session_state.sel_god
    st.session_state.modelo_ia = None 

seleccion = st.sidebar.selectbox("Activo:", list(activos_config.keys()), 
                                 index=list(activos_config.keys()).index(st.session_state.activo_god), 
                                 key="sel_god", on_change=actualizar)
datos_activo = activos_config[seleccion]

# 1. HE AÑADIDO '1h' DE VUELTA AQUÍ
intervalo = st.sidebar.select_slider("Temporalidad Operativa:", options=["1m", "5m", "15m", "1h"], value="1h")

umbral = st.sidebar.slider("Confianza Mínima (%)", 50, 95, 60)
lookback = 30
vigilancia = st.sidebar.checkbox("🚨 Auto-Escaneo God Mode", value=False)

if not HAS_TF: st.error("Instala TensorFlow."); st.stop()

# --- 1. MOTOR DE DATOS (PARCHEADO) ---
def obtener_datos(config, interval, limite=100):
    # A. CCXT
    if config['tipo'] == 'crypto' and HAS_CCXT:
        try:
            exchange = ccxt.kraken()
            ohlcv = exchange.fetch_ohlcv(config['ticker_ccxt'], interval, limit=limite)
            df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df.set_index('Datetime', inplace=True)
            df.index = df.index.tz_localize('UTC').tz_convert(None)
            return df
        except: pass

    # B. YAHOO (CON LIMPIEZA FORZADA DE COLUMNAS)
    try:
        per = "7d" if interval == "1m" else ("60d" if interval == "5m" else "1y")
        if limite > 500: per = "max" # Training mode
        
        # Auto-adjust=True ayuda a limpiar splits
        df = yf.download(config['ticker_y'], period=per, interval=interval, progress=False, auto_adjust=True)
        
        # 1. Aplanar MultiIndex si existe
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 2. Renombrar Forzosamente para evitar KeyError 'Close'
        # Yahoo a veces devuelve: Open, High, Low, Close, Volume, Dividends...
        # Nos quedamos con las primeras 5 columnas que son las OHLCV
        if len(df.columns) >= 5:
            df = df.iloc[:, :5] # Tomar solo las primeras 5
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume'] # Renombrar estándar
        
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        # Limpieza de velas planas
        df = df[df['High'] != df['Low']]
        
        return df
    except Exception as e:
        # st.error(f"Error Yahoo: {e}") # Descomentar para debug
        return pd.DataFrame()

# --- 2. PROCESAMIENTO ---
def procesar_datos(df):
    if len(df) < 50: return pd.DataFrame()
    
    # Copia de seguridad
    df = df.copy()
    
    # Verificar que existe Close
    if 'Close' not in df.columns: return pd.DataFrame()
    
    df['RSI'] = df.ta.rsi(length=14)
    
    try:
        stoch = df.ta.stochrsi(length=14)
        if stoch is not None and not stoch.empty:
            df['Stoch_K'] = stoch.iloc[:, 0]
        else:
            df['Stoch_K'] = 50
    except: df['Stoch_K'] = 50

    df['EMA_Fast'] = df.ta.ema(length=9)
    df['EMA_Slow'] = df.ta.ema(length=50)
    df['ATR'] = df.ta.atr(length=14)
    
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# --- 3. FUNCIONES IA ---
def crear_secuencias(data, target, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(target[i])
    return np.array(X), np.array(y)

def entrenar_cerebro():
    with st.spinner("🧠 Entrenando IA Maestra... (Un momento)"):
        df_train = obtener_datos(datos_activo, intervalo, limite=1500)
        
        if df_train.empty:
            st.warning("⚠️ Error descargando datos. Intentando backup...")
            df_train = obtener_datos(datos_activo, intervalo, limite=300)
            
        if df_train.empty:
            st.error("❌ No hay datos. Yahoo Finance puede estar bloqueando temporalmente.")
            st.stop()
            
        df_proc = procesar_datos(df_train)
        if df_proc.empty: st.error("❌ Datos vacíos tras limpieza."); st.stop()

        features = ['Close', 'RSI', 'Stoch_K', 'EMA_Fast', 'ATR']
        features = [f for f in features if f in df_proc.columns]
        
        try:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df_proc[features].values)
            X, y = crear_secuencias(scaled_data[:-1], df_proc['Target'].values[:-1], lookback)
            
            if len(X) < 10: st.error("❌ Insuficientes secuencias."); st.stop()

            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(lookback, len(features))))
            model.add(Dropout(0.2))
            model.add(LSTM(50))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            
            st.session_state.modelo_ia = model
            st.session_state.scaler_ia = scaler
            st.session_state.features_ia = features
            st.rerun()
            
        except Exception as e:
            st.error(f"Error Training: {e}")
            st.stop()

def predecir_ia():
    df_live = obtener_datos(datos_activo, intervalo, limite=100)
    if df_live.empty: return None, 0.0, pd.DataFrame()
    
    df_proc = procesar_datos(df_live)
    if df_proc.empty or len(df_proc) < lookback: return None, 0.0, pd.DataFrame()
    
    model = st.session_state.modelo_ia
    scaler = st.session_state.scaler_ia
    features = st.session_state.features_ia
    
    missing = [f for f in features if f not in df_proc.columns]
    if missing: return None, 0.0, df_proc
    
    try:
        last_data = df_proc[features].iloc[-lookback:].values
        last_scaled = scaler.transform(last_data)
        X_pred = last_scaled.reshape(1, lookback, len(features))
        
        prediction = model.predict(X_pred, verbose=0)[0][0]
        clase = 1 if prediction > 0.5 else 0
        conf = prediction if clase == 1 else (1 - prediction)
        return clase, conf, df_proc
    except: return None, 0.0, df_proc

# --- 4. TENDENCIA MADRE ---
def analizar_tendencia_madre(config):
    df_h1 = obtener_datos(config, "1h", limite=200)
    if df_h1.empty or len(df_h1) < 50: return "NEUTRO"
    
    df_h1['EMA_50'] = df_h1.ta.ema(length=50)
    df_h1['EMA_200'] = df_h1.ta.ema(length=200)
    df_h1.dropna(inplace=True)
    
    if df_h1.empty: return "NEUTRO"

    ultimo = df_h1.iloc[-1]
    if ultimo['Close'] > ultimo['EMA_200']:
        return "ALCISTA 🚀" if ultimo['EMA_50'] > ultimo['EMA_200'] else "ALCISTA DÉBIL ↗️"
    else:
        return "BAJISTA 🩸" if ultimo['EMA_50'] < ultimo['EMA_200'] else "BAJISTA DÉBIL ↘️"

# --- INTERFAZ ---
col_reset, col_info = st.columns([1,3])
if col_reset.button("🔄 Reset IA"): st.session_state.modelo_ia = None; st.rerun()

placeholder = st.empty()

with placeholder.container():
    if st.session_state.modelo_ia is None:
        entrenar_cerebro()
    else:
        pred_ia, conf_ia, df_micro = predecir_ia()
        tendencia_madre = analizar_tendencia_madre(datos_activo)
        
        # CHEQUEO DE SEGURIDAD PARA LA UI
        if df_micro is not None and not df_micro.empty and 'Close' in df_micro.columns:
            precio = df_micro['Close'].iloc[-1]
            stoch = df_micro['Stoch_K'].iloc[-1] if 'Stoch_K' in df_micro.columns else 50
            
            decision = "ESPERAR"
            color = "#e9ecef"; txt = "#333"; icono = "⏳"
            razon = "Escaneando..."
            
            if conf_ia * 100 >= umbral:
                if pred_ia == 1:
                    if "ALCISTA" in tendencia_madre:
                        decision = "ENTRADA CALL 💎"
                        razon = "✅ IA Alcista + Tendencia H1 Alcista"
                        color = "#d1e7dd"; txt = "#0f5132"; icono = "🚀"
                        if stoch < 20: decision += " [SNIPER]"
                    else:
                        razon = "⚠️ IA Compra vs H1 Bajista"
                else:
                    if "BAJISTA" in tendencia_madre:
                        decision = "ENTRADA PUT 💎"
                        razon = "✅ IA Bajista + Tendencia H1 Bajista"
                        color = "#f8d7da"; txt = "#842029"; icono = "📉"
                        if stoch > 80: decision += " [SNIPER]"
                    else:
                        razon = "⚠️ IA Venta vs H1 Alcista"
            else:
                razon = f"Confianza IA: {conf_ia:.1%}"

            # Tiempos
            now = datetime.now(tz_bolivia)
            if intervalo == "1m": resto = 60 - now.second
            elif intervalo == "5m": resto = (5 - (now.minute % 5)) * 60 - now.second
            elif intervalo == "15m": resto = (15 - (now.minute % 15)) * 60 - now.second
            else: resto = (60 - now.minute) * 60 - now.second # H1
            
            c1, c2, c3 = st.columns([1.5, 1, 1])
            c1.markdown(f"### {seleccion}")
            c1.markdown(f"<h1 style='margin:0'>${precio:.4f}</h1>", unsafe_allow_html=True)
            c2.metric("Tendencia H1", tendencia_madre)
            c3.metric("Cierre Vela", f"{resto}s")
            
            st.markdown(f"""
            <div style="background-color: {color}; padding: 15px; border-radius: 12px; border: 2px solid {txt}; text-align: center; margin-top: 10px;">
                <h1 style="color: {txt}; margin:0;">{icono} {decision}</h1>
                <h3 style="color: {txt}; margin:5px;">{razon}</h3>
                <p style="color: {txt}; font-size: 14px;">IA Confianza: {conf_ia:.1%} | Stoch: {stoch:.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            fig = go.Figure(data=[go.Candlestick(x=df_micro.tail(50).index, open=df_micro.tail(50)['Open'], high=df_micro.tail(50)['High'], low=df_micro.tail(50)['Low'], close=df_micro.tail(50)['Close'])])
            fig.add_trace(go.Scatter(x=df_micro.tail(50).index, y=df_micro.tail(50)['EMA_Slow'], line=dict(color='blue', width=1), name="EMA 50"))
            fig.update_layout(height=350, template="plotly_white", margin=dict(t=10,b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Recibiendo datos... Si esto persiste, cambia de activo o temporalidad.")

if vigilancia:
    time.sleep(3 if intervalo == "1m" else 10)
    st.rerun()
                      
