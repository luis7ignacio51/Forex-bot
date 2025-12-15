import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import pytz
import time
import numpy as np

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Binary Options AI", layout="wide", page_icon="🎰")
st.title("🎰 Binary Options Master (Predicción de Vela H1)")

tz_bolivia = pytz.timezone('America/La_Paz')

# --- DEFINICIÓN DE ACTIVOS ---
pares = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "Bitcoin (BTC)": "BTC-USD",
    "Solana (SOL)": "SOL-USD"
}

# --- GESTIÓN DE MEMORIA ---
if 'activo_binaria' not in st.session_state:
    st.session_state.activo_binaria = "EUR/USD"

if st.session_state.activo_binaria not in pares:
    st.session_state.activo_binaria = list(pares.keys())[0]

# --- SIDEBAR ---
st.sidebar.header("⏱️ Configuración Binaria")

def actualizar_seleccion():
    st.session_state.activo_binaria = st.session_state.selector_binaria

seleccion = st.sidebar.selectbox(
    "Activo a Operar:", 
    list(pares.keys()), 
    index=list(pares.keys()).index(st.session_state.activo_binaria),
    key="selector_binaria",
    on_change=actualizar_seleccion
)
ticker_actual = pares[seleccion]

# Umbral de confianza (Para Binarias, queremos seguridad)
umbral = st.sidebar.slider("Confianza Mínima para Señal (%)", 55, 90, 60)
vigilancia = st.sidebar.checkbox("🚨 Auto-Escaneo (Cada 60s)", value=False)

# --- 1. MOTOR DE DATOS (AUTO-DETECTIVE) ---
@st.cache_data(ttl=60) 
def cargar_datos_binarios(ticker):
    df_final = pd.DataFrame()
    nombre_limpio = ticker.replace("=X","").replace("=F","")
    
    posibles_archivos = [
        f"{nombre_limpio}.csv", "EURUSD.csv", "GBPUSD.csv", "USDJPY.csv", 
        "SOL-USD.csv", "BTCUSD_1h_Combined_Index.csv"
    ]
    
    # Intentar cargar CSV Histórico
    for archivo in posibles_archivos:
        try:
            df_hist = pd.read_csv(archivo, sep=None, engine='python')
            df_hist.columns = df_hist.columns.str.replace('<', '').str.replace('>', '').str.capitalize()
            
            # Detección de Fecha
            if 'Date' in df_hist.columns and 'Time' in df_hist.columns:
                fechas = df_hist['Date'].astype(str).str.replace('.', '-')
                df_hist['Datetime'] = pd.to_datetime(fechas + ' ' + df_hist['Time'].astype(str))
                df_hist.set_index('Datetime', inplace=True)
            else:
                col_fecha = next((c for c in df_hist.columns if 'date' in c.lower() or 'time' in c.lower()), None)
                if col_fecha:
                    df_hist['Datetime'] = pd.to_datetime(df_hist[col_fecha])
                    df_hist.set_index('Datetime', inplace=True)

            cols_req = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df_hist.columns]
            df_final = df_hist[cols_req]
            
            for c in df_final.columns:
                df_final[c] = pd.to_numeric(df_final[c], errors='coerce')
                
            df_final = df_final[df_final.index <= datetime.now()]
            break 
        except:
            continue

    # Relleno Yahoo
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

# --- 2. INGENIERÍA DE CARACTERÍSTICAS (ESTRATEGIA BINARIA) ---
def procesar_binarias(df):
    if df.empty: return df
    
    # A. ACCIÓN DEL PRECIO (Mechas y Cuerpos)
    # Las binarias se ganan detectando rechazos
    df['Cuerpo'] = abs(df['Close'] - df['Open'])
    df['Mecha_Sup'] = df['High'] - np.maximum(df['Close'], df['Open'])
    df['Mecha_Inf'] = np.minimum(df['Close'], df['Open']) - df['Low']
    
    # B. OSCILADORES DE REVERSIÓN (Clave para binarias)
    df['RSI'] = df.ta.rsi(length=14)
    df['CCI'] = df.ta.cci(length=14) # Commodity Channel Index
    df['WillR'] = df.ta.willr(length=14) # Williams %R
    
    # C. TENDENCIA Y VOLATILIDAD
    df['EMA_50'] = df.ta.ema(length=50)
    df['ATR'] = df.ta.atr(length=14)
    
    # Bandas Bollinger (Para rebotes)
    bb = df.ta.bbands(length=20, std=2)
    # Distancia al borde superior e inferior
    if bb is not None:
        df['Dist_BBU'] = bb.iloc[:, 1] - df['Close'] # Upper
        df['Dist_BBL'] = df['Close'] - bb.iloc[:, 0] # Lower

    # D. CONTEXTO
    df['Hora'] = df.index.hour
    
    # E. TARGET: EL COLOR DE LA SIGUIENTE VELA
    # 1 = Verde (Call/Sube), 0 = Roja (Put/Baja)
    df['Target'] = (df['Close'].shift(-1) > df['Open'].shift(-1)).astype(int)
    
    df.dropna(inplace=True)
    return df

# --- 3. CEREBRO IA ---
def ejecutar_ia_binaria(df):
    # Usamos indicadores específicos de reversión y fuerza
    features = ['RSI', 'CCI', 'WillR', 'Mecha_Sup', 'Mecha_Inf', 'Cuerpo', 'EMA_50', 'Dist_BBU', 'Dist_BBL', 'Hora']
    features = [f for f in features if f in df.columns]
    
    # Entrenar con todo el historial disponible (Max precisión)
    # Para binarias, los patrones repetitivos son clave
    train = df.iloc[:-1] # Todo menos la vela actual (que no ha cerrado)
    
    # Random Forest optimizado para clasificación binaria pura
    model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(train[features], train["Target"])
    
    # Predecir la vela actual (que se está formando) para saber cómo cerrará la SIGUIENTE
    ultimo_dato = df.iloc[-1:][features]
    prediccion = model.predict(ultimo_dato)[0]
    probabilidad = model.predict_proba(ultimo_dato)[0]
    
    return prediccion, max(probabilidad), features

# --- 4. INTERFAZ ---
if st.sidebar.button("Forzar Análisis"): st.cache_data.clear()

placeholder = st.empty()

with placeholder.container():
    df_raw = cargar_datos_binarios(ticker_actual)

    if not df_raw.empty:
        df = procesar_binarias(df_raw)
        pred, conf, feats = ejecutar_ia_binaria(df)
        
        # Datos Actuales
        precio = df['Close'].iloc[-1]
        hora_cierre = (df.index[-1] + timedelta(hours=1)).strftime("%H:%M")
        
        # --- LÓGICA DE SEÑAL ---
        decision = "NEUTRO"
        color_card = "#e9ecef"
        color_txt = "#495057"
        icono = "⏳"
        
        # Solo damos señal si supera el umbral de confianza
        if conf * 100 >= umbral:
            if pred == 1:
                decision = "CALL (ALZA) 🟢"
                color_card = "#d4edda" # Verde
                color_txt = "#155724"
                icono = "📈"
            else:
                decision = "PUT (BAJA) 🔴"
                color_card = "#f8d7da" # Rojo
                color_txt = "#721c24"
                icono = "📉"
        else:
            decision = f"NO ENTRAR (Riesgo Alto)"
            icono = "✋"
        
        # HEADER
        c1, c2 = st.columns([2,1])
        c1.markdown(f"### {seleccion} | Precio: **{precio:.5f}**")
        c2.caption(f"Cierre de vela: {hora_cierre}")

        # TARJETA DE SEÑAL GIGANTE
        st.markdown(f"""
        <div style="background-color: {color_card}; padding: 30px; border-radius: 15px; text-align: center; border: 2px solid {color_txt}; margin-bottom: 20px;">
            <h1 style="color: {color_txt}; margin:0; font-size: 50px;">{icono}</h1>
            <h2 style="color: {color_txt}; margin:0;">{decision}</h2>
            <hr style="opacity: 0.3">
            <h4 style="color: {color_txt}; margin:0;">Probabilidad IA: <b>{conf:.1%}</b></h4>
            <p style="color: {color_txt}; margin:0;">Expiración: 1 Hora (Fin de vela)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # DETALLES TÉCNICOS (Para que confíes en la señal)
        with st.expander("📊 Ver Análisis Detallado (Indicadores)"):
            c_i1, c_i2, c_i3, c_i4 = st.columns(4)
            c_i1.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
            c_i2.metric("CCI (Momentum)", f"{df['CCI'].iloc[-1]:.1f}")
            c_i3.metric("Mecha Sup.", f"{df['Mecha_Sup'].iloc[-1]:.5f}")
            c_i4.metric("Mecha Inf.", f"{df['Mecha_Inf'].iloc[-1]:.5f}")
            st.caption("*Mechas largas indican fuerte rechazo del precio.")

        # GRÁFICO
        df_ver = df.tail(60)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_ver.index, open=df_ver['Open'], high=df_ver['High'], low=df_ver['Low'], close=df_ver['Close'], name='Precio'))
        # Bandas Bollinger (Visualmente útiles para binarias)
        if 'BBU_20_2.0' in df.columns:
            fig.add_trace(go.Scatter(x=df_ver.index, y=df_ver['BBU_20_2.0'], line=dict(color='gray', width=1, dash='dot'), name="Banda Sup"))
            fig.add_trace(go.Scatter(x=df_ver.index, y=df_ver['BBL_20_2.0'], line=dict(color='gray', width=1, dash='dot'), name="Banda Inf"))
            
        fig.update_layout(template="plotly_white", height=350, xaxis_rangeslider_visible=False, margin=dict(t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Esperando datos... Verifica tu conexión o archivos.")

if vigilancia:
    time.sleep(60)
    st.rerun()
    
