import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import pytz
import time
import numpy as np

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Forex Sniper Universal", layout="wide", page_icon="🌎")
st.title("🌎 Forex Sniper Universal (Auto-Detect Format)")

tz_bolivia = pytz.timezone('America/La_Paz')

# --- 1. DEFINICIÓN DE PARES ---
pares = {
    "GBP/USD (Libra)": "GBPUSD=X",
    "EUR/USD (Euro)": "EURUSD=X",
    "USD/JPY (Yen)": "JPY=X",
    "AUD/USD (Australiano)": "AUDUSD=X",
    "USD/CAD (Canadiense)": "CAD=X",
    "XAU/USD (Oro)": "GC=F"
}

# --- 2. GESTIÓN DE MEMORIA ---
if 'activo_forex' not in st.session_state:
    st.session_state.activo_forex = "USD/JPY (Yen)"

if st.session_state.activo_forex not in pares:
    st.session_state.activo_forex = list(pares.keys())[0]

# --- BARRA LATERAL ---
st.sidebar.header("🏦 Sala de Operaciones")

def actualizar_seleccion():
    st.session_state.activo_forex = st.session_state.selector_forex

seleccion = st.sidebar.selectbox(
    "Par de Divisas:", 
    list(pares.keys()), 
    index=list(pares.keys()).index(st.session_state.activo_forex),
    key="selector_forex",
    on_change=actualizar_seleccion
)
ticker_actual = pares[seleccion]

vigilancia = st.sidebar.checkbox("🚨 Activar Radar Institucional", value=False)
frecuencia = st.sidebar.slider("Frecuencia (seg)", 10, 300, 60)

# --- 3. MOTOR DE DATOS (AUTO-DETECTIVE) ---
@st.cache_data(ttl=60) 
def cargar_datos_forex(ticker):
    df_final = pd.DataFrame()
    nombre_limpio = ticker.replace("=X","").replace("=F","")
    
    posibles_archivos = [
        f"{nombre_limpio}.csv",              
        f"USD{nombre_limpio}.csv",           
        f"{nombre_limpio}USD.csv",           
        "USDJPY.csv",                        
        "GBPUSD.csv",
        "EURUSD.csv"
    ]
    
    # A. INTENTAR CARGAR CSV HISTÓRICO
    for archivo in posibles_archivos:
        try:
            # MAGIA AQUÍ: sep=None y engine='python' detecta automáticamente si es COMAS o TABS
            df_hist = pd.read_csv(archivo, sep=None, engine='python')
            
            # Limpieza de nombres sucios de MetaTrader (<OPEN> -> Open)
            df_hist.columns = df_hist.columns.str.replace('<', '').str.replace('>', '').str.capitalize()
            
            # CASO 1: Fecha y Hora separadas (Date + Time)
            if 'Date' in df_hist.columns and 'Time' in df_hist.columns:
                # Aseguramos que sean strings para poder sumarlos
                fechas = df_hist['Date'].astype(str)
                horas = df_hist['Time'].astype(str)
                
                # Reemplazamos puntos por guiones (2010.01.01 -> 2010-01-01) para que Pandas entienda
                fechas = fechas.str.replace('.', '-')
                
                df_hist['Datetime'] = pd.to_datetime(fechas + ' ' + horas)
                df_hist.set_index('Datetime', inplace=True)
            
            # CASO 2: Fecha junta
            else:
                col_fecha = next((c for c in df_hist.columns if 'date' in c.lower() or 'time' in c.lower()), None)
                if col_fecha:
                    df_hist['Datetime'] = pd.to_datetime(df_hist[col_fecha])
                    df_hist.set_index('Datetime', inplace=True)

            # Normalizar columnas
            cols_req = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df_hist.columns]
            df_final = df_hist[cols_req]
            
            # Asegurar números
            for c in df_final.columns:
                df_final[c] = pd.to_numeric(df_final[c], errors='coerce')
                
            df_final = df_final[df_final.index <= datetime.now()]
            break 
            
        except Exception:
            continue

    # B. YAHOO FINANCE
    try:
        df_yahoo = yf.download(ticker, period="2y", interval="1h")
        if isinstance(df_yahoo.columns, pd.MultiIndex):
            df_yahoo.columns = df_yahoo.columns.get_level_values(0)
        df_yahoo.index = pd.to_datetime(df_yahoo.index).tz_localize(None)
        
        if not df_final.empty:
            if df_final.index.tz is not None: df_final.index = df_final.index.tz_localize(None)
            df_final = pd.concat([df_final, df_yahoo])
            df_final = df_final[~df_final.index.duplicated(keep='last')]
        else:
            df_final = df_yahoo     
    except:
        pass

    if not df_final.empty: 
        df_final.sort_index(inplace=True)
        df_final.dropna(inplace=True)
        
    return df_final

# --- 4. PROCESAMIENTO ---
def procesar_forex(df):
    if df.empty: return df
    
    df['EMA_50'] = df.ta.ema(length=50)
    df['EMA_200'] = df.ta.ema(length=200)
    df['RSI'] = df.ta.rsi(length=14)
    df['ATR'] = df.ta.atr(length=14)
    
    adx = df.ta.adx(length=14)
    if adx is not None and not adx.empty:
        df['ADX'] = adx.iloc[:, 0]

    df['Hora'] = df.index.hour
    cond_volumen = (df['Hora'] >= 7) & (df['Hora'] <= 20)
    df['Sesion_Activa'] = np.where(cond_volumen, 1, 0)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.dropna(inplace=True)
    return df

# --- 5. IA ---
def ejecutar_ia_forex(df):
    features = ['RSI', 'EMA_50', 'EMA_200', 'ATR', 'ADX', 'Sesion_Activa', 'Hora']
    features = [f for f in features if f in df.columns]
    
    limit_train = 5000 if len(df) > 10000 else 1000
    train = df.iloc[-limit_train:] 
    
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(train[features], train["Target"])
    
    ultimo_dato = df.iloc[-1:][features]
    prediccion = model.predict(ultimo_dato)[0]
    probabilidad = model.predict_proba(ultimo_dato)[0]
    
    return prediccion, max(probabilidad), features

# --- 6. INTERFAZ ---
if st.sidebar.button("Forzar Recarga"): st.cache_data.clear()

placeholder = st.empty()

with placeholder.container():
    df_raw = cargar_datos_forex(ticker_actual)

    if not df_raw.empty:
        df = procesar_forex(df_raw)
        pred, conf, feats = ejecutar_ia_forex(df)
        
        precio = df['Close'].iloc[-1]
        adx_val = df['ADX'].iloc[-1] if 'ADX' in df.columns else 0
        ema_50 = df['EMA_50'].iloc[-1]
        atr_val = df['ATR'].iloc[-1]
        tendencia = "ALCISTA" if precio > df['EMA_200'].iloc[-1] else "BAJISTA"
        
        decision = "NEUTRO"
        color_box = "#e2e3e5"
        text_color = "#383d41"
        
        if adx_val > 20:
            if tendencia == "ALCISTA" and pred == 1:
                decision = "LONG (COMPRA) 🚀"
                color_box = "#d4edda"
                text_color = "#155724"
            elif tendencia == "BAJISTA" and pred == 0:
                decision = "SHORT (VENTA) 📉"
                color_box = "#f8d7da"
                text_color = "#721c24"
            else:
                decision = "ESPERAR (Divergencia) ✋"
                color_box = "#fff3cd"
                text_color = "#856404"
        else:
            decision = "MERCADO LATERAL (ADX Bajo) 💤"
        
        hora_local = datetime.now(tz_bolivia).strftime("%H:%M:%S")
        c1, c2 = st.columns([2,1])
        c1.markdown(f"### {seleccion} <span style='font-size:26px'>${precio:,.3f}</span>", unsafe_allow_html=True)
        
        origen = "CSV Histórico + Yahoo" if len(df) > 20000 else "Yahoo Finance (Reciente)"
        c2.caption(f"Fuente: {origen} | {hora_local}")

        st.markdown(f"""
        <div style="background-color: {color_box}; padding: 20px; border-radius: 10px; border-left: 6px solid {text_color}; margin-bottom: 20px;">
            <h2 style="color: {text_color}; margin:0;">{decision}</h2>
            <p style="color: {text_color}; margin:0;">
                IA Confianza: <b>{conf:.1%}</b> | Fuerza (ADX): <b>{adx_val:.1f}</b> | Tendencia: <b>{tendencia}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        if "LONG" in decision or "SHORT" in decision:
            c_ent, c_sl, c_tp = st.columns(3)
            c_ent.info(f"📍 Entrada (EMA50): {ema_50:.3f}")
            
            # Ajuste de decimales para JPY (3) vs otros (5)
            digits = 3 if "JPY" in ticker_actual else 5
            
            factor = 1.5
            if "LONG" in decision:
                sl = precio - (atr_val * factor)
                tp = precio + (atr_val * (factor * 2))
            else:
                sl = precio + (atr_val * factor)
                tp = precio - (atr_val * (factor * 2))
                
            c_sl.error(f"🛑 Stop Loss: {sl:.{digits}f}")
            c_tp.success(f"💰 Take Profit: {tp:.{digits}f}")

        df_ver = df.tail(120)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_ver.index, open=df_ver['Open'], high=df_ver['High'], low=df_ver['Low'], close=df_ver['Close'], name='Precio'))
        fig.add_trace(go.Scatter(x=df_ver.index, y=df_ver['EMA_50'], line=dict(color='cyan', width=1), name="EMA 50"))
        fig.add_trace(go.Scatter(x=df_ver.index, y=df_ver['EMA_200'], line=dict(color='blue', width=2), name="EMA 200"))
        fig.update_layout(template="plotly_white", height=400, xaxis_rangeslider_visible=False, margin=dict(t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("No se encontraron datos. Asegúrate de haber subido USDJPY.csv a GitHub.")

if vigilancia:
    time.sleep(frecuencia)
    st.rerun()
    
