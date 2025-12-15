import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from datetime import datetime, timedelta
import pytz
import time
import numpy as np

# --- IMPORTACIÓN SEGURA DE CCXT ---
try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Transparent AI Scalper", layout="wide", page_icon="👁️")
st.title("👁️ Transparent AI Scalper (Diagnóstico Total)")

tz_bolivia = pytz.timezone('America/La_Paz')

# --- 1. CONFIGURACIÓN DE ACTIVOS ---
activos_config = {
    "Bitcoin (BTC)": {"tipo": "crypto", "ticker_ccxt": "BTC/USD", "ticker_y": "BTC-USD"},
    "Solana (SOL)":  {"tipo": "crypto", "ticker_ccxt": "SOL/USD", "ticker_y": "SOL-USD"},
    "Ethereum (ETH)": {"tipo": "crypto", "ticker_ccxt": "ETH/USD", "ticker_y": "ETH-USD"},
    "EUR/USD":       {"tipo": "forex",  "ticker_ccxt": None,      "ticker_y": "EURUSD=X"},
    "GBP/USD":       {"tipo": "forex",  "ticker_ccxt": None,      "ticker_y": "GBPUSD=X"},
    "USD/JPY":       {"tipo": "forex",  "ticker_ccxt": None,      "ticker_y": "JPY=X"},
    "Gold (XAU)":    {"tipo": "forex",  "ticker_ccxt": None,      "ticker_y": "GC=F"}
}

# --- SIDEBAR ---
st.sidebar.header("🎛️ Panel de Control")
if 'activo_turbo' not in st.session_state: st.session_state.activo_turbo = "Solana (SOL)"
def actualizar(): st.session_state.activo_turbo = st.session_state.sel_turbo

seleccion = st.sidebar.selectbox("Activo:", list(activos_config.keys()), 
                                 index=list(activos_config.keys()).index(st.session_state.activo_turbo), 
                                 key="sel_turbo", on_change=actualizar)

datos_activo = activos_config[seleccion]
intervalo = st.sidebar.select_slider("Temporalidad:", options=["1m", "5m", "15m", "1h"], value="1h")
umbral = st.sidebar.slider("Confianza Mínima IA (%)", 60, 95, 75)
vigilancia = st.sidebar.checkbox("🚨 Auto-Escaneo (Live)", value=False)

# --- 2. MOTOR DE DATOS ---
@st.cache_data(ttl=10 if intervalo == "1m" else 30)
def obtener_datos_turbo(config, interval):
    # A. CCXT (Cripto)
    if config['tipo'] == 'crypto' and HAS_CCXT:
        try:
            exchange = ccxt.kraken() 
            ohlcv = exchange.fetch_ohlcv(config['ticker_ccxt'], interval, limit=100)
            df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df.set_index('Datetime', inplace=True)
            df.index = df.index.tz_localize('UTC').tz_convert(None) 
            return df[['Open', 'High', 'Low', 'Close', 'Volume']], "CCXT (Kraken)"
        except: pass

    # B. YAHOO (Forex/Backup)
    try:
        y_per = "1d" if interval == "1m" else ("5d" if interval == "5m" else "1mo")
        df = yf.download(config['ticker_y'], period=y_per, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[df['High'] != df['Low']] # Limpieza velas fantasmas
        return df, "Yahoo Finance"
    except:
        return pd.DataFrame(), "Error"

# --- 3. PROCESAMIENTO ---
def procesar_indicadores(df):
    if len(df) < 20: return pd.DataFrame()
    
    df['EMA_Fast'] = df.ta.ema(length=7)
    df['EMA_Slow'] = df.ta.ema(length=25)
    try: df['EMA_Trend'] = df.ta.ema(length=100) 
    except: df['EMA_Trend'] = df['EMA_Slow']
        
    df['RSI'] = df.ta.rsi(length=14)
    df['CCI'] = df.ta.cci(length=14)
    
    bb = df.ta.bbands(length=20, std=2)
    if bb is not None:
        df['BB_Up'] = bb.iloc[:, 1]
        df['BB_Low'] = bb.iloc[:, 0]
        
    cuerpo = abs(df['Close'] - df['Open'])
    mecha_sup = df['High'] - np.maximum(df['Close'], df['Open'])
    mecha_inf = np.minimum(df['Close'], df['Open']) - df['Low']
    df['Pinbar'] = np.where((mecha_inf > cuerpo*2) | (mecha_sup > cuerpo*2), 1, 0)
    
    df['Target'] = (df['Close'].shift(-1) > df['Open'].shift(-1)).astype(int)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# --- 4. INTELIGENCIA ARTIFICIAL ---
def consultar_oraculo(df):
    features = ['RSI', 'CCI', 'EMA_Fast', 'EMA_Slow', 'BB_Up', 'BB_Low', 'Pinbar']
    features = [f for f in features if f in df.columns]
    
    if len(df) < 30: return 0, 0.0 
    
    train = df.iloc[:-1]
    try:
        model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(train[features], train["Target"])
        
        ultimo = df.iloc[-1:][features]
        pred = model.predict(ultimo)[0]
        prob = model.predict_proba(ultimo)[0]
        confianza = max(prob)
        
        if confianza > 0.999: return 0, 0.0 # Anti-Alucinación
        return pred, confianza
    except: return 0, 0.0

# --- 5. INTERFAZ VISUAL ---
if st.sidebar.button("⚡ Forzar Recarga"): st.cache_data.clear()
placeholder = st.empty()

with placeholder.container():
    df_live, fuente = obtener_datos_turbo(datos_activo, intervalo)
    
    if not df_live.empty and len(df_live) > 20:
        df = procesar_indicadores(df_live)
        
        if not df.empty:
            pred, conf = consultar_oraculo(df)
            
            # Variables actuales
            precio = df['Close'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            trend = df['EMA_Trend'].iloc[-1]
            
            # --- LÓGICA DE DIAGNÓSTICO DETALLADA ---
            decision = "ESPERAR"
            icono = "⏳"
            color = "#e9ecef"; txt = "#333"
            
            # 1. ¿Tiene la IA suficiente confianza?
            if conf*100 < umbral:
                razon = f"Confianza IA baja ({conf:.1%} < {umbral}%)"
                
            else:
                # 2. IA dice COMPRA (1)
                if pred == 1:
                    # Filtros
                    if precio > trend:
                        decision = "CALL (SUBIR) 🚀"
                        razon = "✅ IA Alcista + Tendencia a favor"
                        color = "#d4edda"; txt = "#155724"; icono = "📈"
                    elif rsi < 30:
                        decision = "CALL (REBOTE) 🚀"
                        razon = "✅ IA Alcista + RSI Sobrevendido"
                        color = "#d4edda"; txt = "#155724"; icono = "📈"
                    else:
                        razon = "⛔ IA Alcista pero Tendencia Bajista y RSI Neutro"
                        
                # 3. IA dice VENTA (0)
                else:
                    # Filtros
                    if precio < trend:
                        decision = "PUT (BAJAR) 📉"
                        razon = "✅ IA Bajista + Tendencia a favor"
                        color = "#f8d7da"; txt = "#721c24"; icono = "📉"
                    elif rsi > 70:
                        decision = "PUT (REBOTE) 📉"
                        razon = "✅ IA Bajista + RSI Sobrecomprado"
                        color = "#f8d7da"; txt = "#721c24"; icono = "📉"
                    else:
                        razon = "⛔ IA Bajista pero Tendencia Alcista y RSI Neutro"
            
            # Tiempo Restante
            now = datetime.now(tz_bolivia)
            if intervalo == "1m": resto = 60 - now.second
            elif intervalo == "5m": resto = (5 - (now.minute % 5)) * 60 - now.second
            else: resto = (60 - now.minute) * 60 - now.second
            
            # UI
            c1, c2 = st.columns([2,1])
            c1.markdown(f"### {seleccion} [{intervalo}]")
            c1.markdown(f"<h1 style='margin:0'>${precio:.5f}</h1>", unsafe_allow_html=True)
            
            if fuente == "CCXT (Kraken)":
                c2.markdown("<span style='color:green; font-weight:bold'>🟢 CONEXIÓN DIRECTA</span>", unsafe_allow_html=True)
            else:
                c2.markdown(f"<span style='color:orange; font-weight:bold'>🟠 {fuente}</span>", unsafe_allow_html=True)
                
            c2.metric("Cierre Vela:", f"{resto}s")
            
            st.markdown(f"""
            <div style="background-color: {color}; padding: 15px; border-radius: 10px; border: 2px solid {txt}; text-align: center;">
                <h1 style="color: {txt}; margin:0;">{icono} {decision}</h1>
                <p style="color: {txt}; margin:0; font-size: 18px;"><b>Motivo: {razon}</b></p>
                <hr style="opacity: 0.2">
                <p style="color: {txt}; font-size:12px;">Confianza IA: {conf:.1%} | RSI: {rsi:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Gráfico
            fig = go.Figure(data=[go.Candlestick(x=df.tail(40).index, open=df.tail(40)['Open'], high=df.tail(40)['High'], low=df.tail(40)['Low'], close=df.tail(40)['Close'])])
            fig.add_trace(go.Scatter(x=df.tail(40).index, y=df.tail(40)['EMA_Fast'], line=dict(color='orange', width=1), name='EMA Rápida'))
            fig.add_trace(go.Scatter(x=df.tail(40).index, y=df.tail(40)['EMA_Trend'], line=dict(color='blue', width=1), name='Tendencia'))
            fig.update_layout(height=350, xaxis_rangeslider_visible=False, margin=dict(t=10,b=0), template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Analizando estructura de mercado... (Pocos datos limpios)")
    else:
        st.error(f"Esperando datos de {fuente}... (Puede tardar unos segundos en conectar)")

if vigilancia:
    time.sleep(5 if intervalo == "1m" else 15)
    st.rerun()
