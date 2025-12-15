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

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Scalping AI Pro", layout="wide", page_icon="⚡")
st.title("⚡ Scalping AI Pro (M1/M5/H1 Multi-Timeframe)")

tz_bolivia = pytz.timezone('America/La_Paz')

# --- SIDEBAR ---
st.sidebar.header("🎛️ Centro de Comando")

# 1. Selector de Activo
pares = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X",
    "Bitcoin (BTC)": "BTC-USD", "Solana (SOL)": "SOL-USD", "Gold (XAU)": "GC=F"
}
if 'activo_scalping' not in st.session_state: st.session_state.activo_scalping = "EUR/USD"
def actualizar(): st.session_state.activo_scalping = st.session_state.sel_scalp
seleccion = st.sidebar.selectbox("Activo:", list(pares.keys()), index=list(pares.keys()).index(st.session_state.activo_scalping), key="sel_scalp", on_change=actualizar)
ticker = pares[seleccion]

# 2. Selector de Temporalidad (NUEVO)
intervalo = st.sidebar.select_slider("Temporalidad (Velas):", options=["1m", "5m", "15m", "1h"], value="1h")

# 3. Ajustes
umbral = st.sidebar.slider("Confianza IA (%)", 55, 95, 70)
vigilancia = st.sidebar.checkbox("🚨 Auto-Escaneo (Live)", value=False)

# --- 1. MOTOR DE DATOS HÍBRIDO ---
@st.cache_data(ttl=30 if intervalo in ['1m','5m'] else 60) 
def cargar_datos_multitimeframe(ticker, interval):
    df_final = pd.DataFrame()
    nombre_limpio = ticker.replace("=X","").replace("=F","")
    
    # ESTRATEGIA DE CARGA:
    # Si es H1 -> Intentamos usar CSV Histórico (Mejor calidad)
    # Si es M1/M5 -> Usamos Yahoo Finance (Datos recientes en vivo, ya que CSV de H1 no sirve)
    
    usar_csv = False
    if interval == "1h":
        archivos = [f"{nombre_limpio}.csv", "EURUSD.csv", "GBPUSD.csv", "USDJPY.csv", "SOL-USD.csv"]
        for f in archivos:
            try:
                df = pd.read_csv(f, sep=None, engine='python')
                df.columns = df.columns.str.replace('<','').str.replace('>','').str.capitalize()
                
                # Procesar fecha
                if 'Date' in df.columns and 'Time' in df.columns:
                    df['Datetime'] = pd.to_datetime(df['Date'].astype(str).str.replace('.','-') + ' ' + df['Time'].astype(str))
                else:
                    col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()), None)
                    if col: df['Datetime'] = pd.to_datetime(df[col])
                
                df.set_index('Datetime', inplace=True)
                cols = [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
                df_final = df[cols]
                for c in df_final.columns: df_final[c] = pd.to_numeric(df_final[c], errors='coerce')
                df_final = df_final[df_final.index <= datetime.now()]
                usar_csv = True
                break
            except: continue

    # YAHOO FINANCE (Fundamental para M1/M5)
    try:
        # Ajustamos el periodo según el intervalo (Yahoo tiene límites)
        per = "7d" if interval == "1m" else ("60d" if interval in ["5m","15m"] else "2y")
        
        df_y = yf.download(ticker, period=per, interval=interval)
        if isinstance(df_y.columns, pd.MultiIndex): df_y.columns = df_y.columns.get_level_values(0)
        df_y.index = pd.to_datetime(df_y.index).tz_localize(None)
        
        if not df_final.empty and usar_csv:
            if df_final.index.tz is not None: df_final.index = df_final.index.tz_localize(None)
            df_final = pd.concat([df_final, df_y])
            df_final = df_final[~df_final.index.duplicated(keep='last')]
        else:
            df_final = df_y
    except: pass
    
    if not df_final.empty: 
        df_final.sort_index(inplace=True)
        df_final.dropna(inplace=True)
        
    return df_final

# --- 2. PROCESAMIENTO ADAPTATIVO ---
def procesar_scalping(df, interval):
    if df.empty: return df
    
    # Ajustamos indicadores según velocidad
    # En M1 necesitamos reacciones más rápidas que en H1
    len_rapida = 9 if interval == "1m" else 20
    len_lenta = 21 if interval == "1m" else 50
    
    df['EMA_Rapida'] = df.ta.ema(length=len_rapida)
    df['EMA_Lenta'] = df.ta.ema(length=len_lenta)
    df['EMA_200'] = df.ta.ema(length=200) # Tendencia Madre
    
    df['RSI'] = df.ta.rsi(length=14)
    df['ATR'] = df.ta.atr(length=14)
    
    # Bandas Bollinger (Rebotes M1)
    bb = df.ta.bbands(length=20, std=2)
    if bb is not None:
        df['BB_Up'] = bb.iloc[:, 1]
        df['BB_Low'] = bb.iloc[:, 0]
        
    # Agotamiento (CCI)
    df['CCI'] = df.ta.cci(length=14)

    # Patrones de Vela (Matemáticos)
    cuerpo = abs(df['Close'] - df['Open'])
    mecha_sup = df['High'] - np.maximum(df['Close'], df['Open'])
    mecha_inf = np.minimum(df['Close'], df['Open']) - df['Low']
    
    # Martillo (Bullish) & Estrella (Bearish)
    df['Patron_Bull'] = np.where((mecha_inf > cuerpo*1.5) & (mecha_sup < cuerpo*0.5), 1, 0)
    df['Patron_Bear'] = np.where((mecha_sup > cuerpo*1.5) & (mecha_inf < cuerpo*0.5), 1, 0)

    # TARGET:
    # En M1/M5, queremos saber si la PRÓXIMA vela es del color opuesto (Reversión) o igual (Continuidad)
    # Aquí entrenamos para PREDECIR COLOR (1=Verde, 0=Roja)
    df['Target'] = (df['Close'].shift(-1) > df['Open'].shift(-1)).astype(int)
    
    df.dropna(inplace=True)
    return df

# --- 3. IA SCALPER ---
def ejecutar_ia_scalper(df):
    features = ['RSI', 'CCI', 'EMA_Rapida', 'EMA_Lenta', 'BB_Up', 'BB_Low', 'Patron_Bull', 'Patron_Bear']
    features = [f for f in features if f in df.columns]
    
    # Entrenar (Usamos menos historia en M1 para adaptarnos al régimen actual)
    limit = 1000 
    train = df.iloc[-limit:-1]
    
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(train[features], train["Target"])
    
    ultimo = df.iloc[-1:][features]
    pred = model.predict(ultimo)[0]
    prob = model.predict_proba(ultimo)[0]
    
    return pred, max(prob)

# --- 4. INTERFAZ VISUAL ---
if st.sidebar.button("Forzar Recarga"): st.cache_data.clear()
placeholder = st.empty()

with placeholder.container():
    df_raw = cargar_datos_multitimeframe(ticker, intervalo)
    
    if not df_raw.empty:
        df = procesar_scalping(df_raw, intervalo)
        pred, conf = ejecutar_ia_scalper(df)
        
        # Variables Clave
        precio = df['Close'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        
        # Lógica de Decisión (Scalping)
        decision = "ESPERAR"
        color_box = "#e9ecef"
        txt_color = "#333"
        icono = "⏳"
        
        # Filtros Estrictos para M1/M5
        if conf*100 >= umbral:
            # CALL: Predicción Verde + (Tendencia Alcista O RSI Sobrevendido)
            if pred == 1:
                if df['Close'].iloc[-1] > df['EMA_Lenta'].iloc[-1] or rsi < 30:
                    decision = "CALL (ALZA) 🚀"
                    color_box = "#d4edda"; txt_color = "#155724"; icono = "📈"
                    
            # PUT: Predicción Roja + (Tendencia Bajista O RSI Sobrecomprado)
            elif pred == 0:
                if df['Close'].iloc[-1] < df['EMA_Lenta'].iloc[-1] or rsi > 70:
                    decision = "PUT (BAJA) 📉"
                    color_box = "#f8d7da"; txt_color = "#721c24"; icono = "📉"
        
        # Cálculo de tiempo restante vela
        now = datetime.now(tz_bolivia)
        minutos_actuales = now.minute
        if intervalo == "1m": resto = 60 - now.second
        elif intervalo == "5m": resto = (5 - (minutos_actuales % 5)) * 60 - now.second
        elif intervalo == "15m": resto = (15 - (minutos_actuales % 15)) * 60 - now.second
        else: resto = (60 - minutes_actuales) * 60 # H1
        
        # Render
        c1, c2 = st.columns([2,1])
        c1.markdown(f"### {seleccion} [{intervalo}]")
        c1.markdown(f"<h1 style='margin:0'>${precio:.5f}</h1>", unsafe_allow_html=True)
        c2.metric("Cierre de Vela en:", f"{resto} seg")
        
        st.markdown(f"""
        <div style="background-color: {color_box}; padding: 20px; border-radius: 15px; border: 3px solid {txt_color}; text-align: center;">
            <h1 style="color: {txt_color}; margin:0; font-size: 40px;">{icono}</h1>
            <h2 style="color: {txt_color}; margin:0;">{decision}</h2>
            <p style="color: {txt_color}; font-weight: bold;">Confianza IA: {conf:.1%} (Meta: {umbral}%)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Gráfico Live
        df_ver = df.tail(60) # Ver últimos 60 periodos (60 mins en M1)
        fig = go.Figure(data=[go.Candlestick(x=df_ver.index, open=df_ver['Open'], high=df_ver['High'], low=df_ver['Low'], close=df_ver['Close'])])
        
        # Indicadores en Gráfico
        fig.add_trace(go.Scatter(x=df_ver.index, y=df_ver['EMA_Rapida'], line=dict(color='orange', width=1), name=f'EMA Rapida'))
        fig.add_trace(go.Scatter(x=df_ver.index, y=df_ver['EMA_Lenta'], line=dict(color='cyan', width=1), name=f'EMA Lenta'))
        
        # Bandas Bollinger (Esenciales para Scalping)
        fig.add_trace(go.Scatter(x=df_ver.index, y=df_ver['BB_Up'], line=dict(color='gray', width=1, dash='dot'), name='BB Sup'))
        fig.add_trace(go.Scatter(x=df_ver.index, y=df_ver['BB_Low'], line=dict(color='gray', width=1, dash='dot'), name='BB Inf'))
        
        fig.update_layout(height=400, template="plotly_white", xaxis_rangeslider_visible=False, margin=dict(t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("Cargando datos... (Si usas M1, Yahoo solo da los últimos 7 días)")

# Refresh ultra-rápido para M1
if vigilancia:
    sleep_time = 10 if intervalo == "1m" else (30 if intervalo == "5m" else 60)
    time.sleep(sleep_time)
    st.rerun()
