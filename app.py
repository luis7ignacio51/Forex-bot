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
st.set_page_config(page_title="Forex Sniper Pro", layout="wide", page_icon="💶")
st.title("💶 Forex Sniper Pro (EUR/USD Institucional)")

tz_bolivia = pytz.timezone('America/La_Paz') # Ajusta a tu zona horaria

# --- MEMORIA DE SESIÓN ---
if 'activo_forex' not in st.session_state:
    st.session_state.activo_forex = "EUR/USD"

# --- BARRA LATERAL ---
st.sidebar.header("🏦 Sala de Operaciones")
pares = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "XAU/USD (Oro)": "GC=F"
}

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

# --- MODO AUTO-TRADING ---
vigilancia = st.sidebar.checkbox("🚨 Activar Radar Institucional (Auto)", value=False)
frecuencia = st.sidebar.slider("Velocidad de Escaneo (seg)", 10, 300, 60)

# --- 1. MOTOR DE DATOS (HÍBRIDO) ---
@st.cache_data(ttl=60) 
def cargar_datos_forex(ticker):
    df_final = pd.DataFrame()
    
    # Nombre de archivo limpio (ej: EURUSD.csv)
    nombre_limpio = ticker.replace("=X","").replace("=F","") + ".csv"
    
    # A. Intentar CSV Histórico (Si consigues uno de 10 años, ponlo aquí)
    try:
        # Intentamos leer archivos comunes
        posibles_nombres = [nombre_limpio, "eurusd_hour.csv", "EURUSD_1h.csv"]
        for f in posibles_nombres:
            try:
                df_hist = pd.read_csv(f)
                # Detección inteligente de fecha
                col_fecha = next((c for c in df_hist.columns if 'date' in c.lower() or 'time' in c.lower()), None)
                if col_fecha: 
                    df_hist['Datetime'] = pd.to_datetime(df_hist[col_fecha])
                    df_hist.set_index('Datetime', inplace=True)
                    # Normalizar columnas
                    df_hist.rename(columns=lambda x: x.capitalize(), inplace=True)
                    # Filtrar columnas necesarias
                    cols_req = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df_hist.columns]
                    df_final = df_hist[cols_req]
                    break # Si funcionó, salimos del bucle
            except:
                continue
    except:
        pass

    # B. Yahoo Finance (Relleno o base)
    try:
        # Descargamos máximo permitido por Yahoo para 1h (730 días)
        df_yahoo = yf.download(ticker, period="2y", interval="1h")
        if isinstance(df_yahoo.columns, pd.MultiIndex):
            df_yahoo.columns = df_yahoo.columns.get_level_values(0)
        df_yahoo.index = pd.to_datetime(df_yahoo.index).tz_localize(None)
        
        if not df_final.empty:
            # Asegurar que ambos índices no tengan zona horaria
            if df_final.index.tz is not None: df_final.index = df_final.index.tz_localize(None)
            
            df_final = pd.concat([df_final, df_yahoo])
            df_final = df_final[~df_final.index.duplicated(keep='last')]
        else:
            df_final = df_yahoo
    except:
        pass

    if not df_final.empty: 
        df_final.sort_index(inplace=True)
        # Filtro de limpieza final
        df_final = df_final[df_final['Close'] > 0]
        
    return df_final

# --- 2. INGENIERÍA DE DATOS FOREX ---
def procesar_forex(df):
    if df.empty: return df
    
    # --- A. Indicadores Técnicos ---
    df['EMA_50'] = df.ta.ema(length=50)
    df['EMA_200'] = df.ta.ema(length=200) # Tendencia Madre
    df['RSI'] = df.ta.rsi(length=14)
    df['ATR'] = df.ta.atr(length=14)      # Volatilidad
    
    # ADX (Fuerza de Tendencia) - CRUCIAL en Forex
    # Si ADX < 20, el mercado está lateral (Rango). No operar.
    adx = df.ta.adx(length=14)
    if adx is not None and not adx.empty:
        df = pd.concat([df, adx], axis=1)
        # Renombrar columna ADX (pandas_ta a veces usa nombres raros)
        col_adx = [c for c in df.columns if 'ADX' in c][0]
        df['ADX'] = df[col_adx]

    # --- B. Filtros de Sesión (Killzones) ---
    # La IA debe saber si es hora de Londres (Volatilidad) o Asia (Calma)
    # Asumimos hora UTC en los datos (Yahoo suele dar UTC)
    df['Hora'] = df.index.hour
    
    # Definimos Sesiones (Aprox UTC):
    # Londres: 07:00 - 16:00
    # Nueva York: 12:00 - 21:00
    # Asia: 00:00 - 09:00
    
    # Creamos una feature "Sesion_Activa" (1 si es Londres/NY, 0 si es Asia/Cierre)
    # Esto enseña a la IA a priorizar señales en horarios de volumen
    cond_londres_ny = (df['Hora'] >= 7) & (df['Hora'] <= 20)
    df['Sesion_Activa'] = np.where(cond_londres_ny, 1, 0)

    # Target
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.dropna(inplace=True)
    return df

# --- 3. CEREBRO IA INSTITUCIONAL ---
def ejecutar_ia_forex(df):
    # Features optimizadas para Forex
    features = ['RSI', 'EMA_50', 'EMA_200', 'ATR', 'ADX', 'Sesion_Activa', 'Hora']
    
    # Verificar que existan todas
    features = [f for f in features if f in df.columns]
    
    # Entrenar (Ventana rodante de 1000 velas para captar el régimen actual)
    train = df.iloc[-1000:] 
    
    # Modelo más robusto
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(train[features], train["Target"])
    
    ultimo_dato = df.iloc[-1:][features]
    prediccion = model.predict(ultimo_dato)[0]
    probabilidad = model.predict_proba(ultimo_dato)[0]
    
    return prediccion, max(probabilidad), model.feature_importances_, features

# --- 4. INTERFAZ ---
if st.sidebar.button("Forzar Análisis Manual"): st.cache_data.clear()

placeholder = st.empty()

with placeholder.container():
    df_raw = cargar_datos_forex(ticker_actual)

    if not df_raw.empty:
        df = procesar_forex(df_raw)
        
        # Ejecutar IA
        pred, conf, importancias, feat_names = ejecutar_ia_forex(df)
        
        # Datos en tiempo real
        precio = df['Close'].iloc[-1]
        adx_val = df['ADX'].iloc[-1]
        rsi_val = df['RSI'].iloc[-1]
        ema_50 = df['EMA_50'].iloc[-1]
        atr_val = df['ATR'].iloc[-1]
        sesion_actual = df['Sesion_Activa'].iloc[-1]
        
        # --- LÓGICA DE FILTRADO (EL SECRETO DEL ÉXITO) ---
        # No operamos si el ADX es muy bajo (mercado muerto)
        filtro_adx = adx_val > 20 
        
        tendencia = "ALCISTA" if precio > df['EMA_200'].iloc[-1] else "BAJISTA"
        
        estado = "NEUTRO"
        color_bg = "#f0f2f6"
        color_txt = "#31333F"
        
        # Reglas de Entrada Sniper
        if filtro_adx:
            if tendencia == "ALCISTA" and pred == 1 and sesion_actual == 1:
                estado = "LONG (COMPRA) 💶"
                color_bg = "#d1e7dd" # Verde
                color_txt = "#0f5132"
            elif tendencia == "BAJISTA" and pred == 0 and sesion_actual == 1:
                estado = "SHORT (VENTA) 📉"
                color_bg = "#f8d7da" # Rojo
                color_txt = "#842029"
            elif sesion_actual == 0:
                estado = "ESPERAR (Volumen Bajo)"
                color_bg = "#fff3cd" # Amarillo
                color_txt = "#664d03"
            else:
                estado = "ESPERAR (Sin Confirmación)"
        else:
            estado = "RANGO / MERCADO LATERAL 💤"
            color_bg = "#e2e3e5" # Gris
            color_txt = "#41464b"

        # --- VISUALIZACIÓN ---
        hora_local = datetime.now(tz_bolivia).strftime("%H:%M:%S")
        
        c1, c2 = st.columns([2, 1])
        c1.markdown(f"### {seleccion} <span style='font-size:28px'>${precio:,.5f}</span>", unsafe_allow_html=True)
        c2.caption(f"Actualizado: {hora_local}")

        # TARJETA DE SEÑAL
        st.markdown(f"""
        <div style="background-color: {color_bg}; padding: 20px; border-radius: 12px; border: 1px solid {color_txt}; margin-bottom: 20px;">
            <h2 style="color: {color_txt}; margin:0; text-align: center;">{estado}</h2>
            <hr style="border-color: {color_txt}; opacity: 0.2;">
            <div style="display: flex; justify-content: space-around; color: {color_txt};">
                <span><b>IA Confianza:</b> {conf:.1%}</span>
                <span><b>ADX (Fuerza):</b> {adx_val:.1f}</span>
                <span><b>Sesión:</b> {'🟢 Activa' if sesion_actual else '🔴 Baja'}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # NIVELES DE ENTRADA Y SALIDA (Solo si hay señal o tendencia clara)
        if "LONG" in estado or "SHORT" in estado:
            c_ent, c_sl, c_tp = st.columns(3)
            
            c_ent.info(f"📍 **Entrada (Pullback):** {ema_50:.5f}")
            
            # SL y TP Institucionales (Ratio 1:1.5 o 1:2)
            if "LONG" in estado:
                sl = precio - (atr_val * 1.5)
                tp = precio + (atr_val * 2.5) # Buscamos recorridos largos
            else:
                sl = precio + (atr_val * 1.5)
                tp = precio - (atr_val * 2.5)
                
            c_sl.error(f"🛑 **Stop Loss:** {sl:.5f}")
            c_tp.success(f"💰 **Take Profit:** {tp:.5f}")

        # GRÁFICO TÉCNICO
        df_ver = df.tail(100)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_ver.index, open=df_ver['Open'], high=df_ver['High'], low=df_ver['Low'], close=df_ver['Close'], name='Precio'))
        fig.add_trace(go.Scatter(x=df_ver.index, y=df_ver['EMA_50'], line=dict(color='cyan', width=1), name="EMA 50"))
        fig.add_trace(go.Scatter(x=df_ver.index, y=df_ver['EMA_200'], line=dict(color='blue', width=2), name="EMA 200"))
        
        fig.update_layout(template="plotly_white", height=400, xaxis_rangeslider_visible=False, margin=dict(t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # EXPLICACIÓN IA
        with st.expander("🧠 ¿Qué está viendo la IA? (Ver Factores)"):
            imp_df = pd.DataFrame({'Factor': feat_names, 'Peso': importancias}).sort_values(by='Peso', ascending=True)
            fig_imp = go.Figure(go.Bar(x=imp_df['Peso'], y=imp_df['Factor'], orientation='h', marker_color='#2E86C1'))
            fig_imp.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_imp)

    else:
        st.warning("Cargando datos de Forex...")

# LOOP
if vigilancia:
    time.sleep(frecuencia)
    st.rerun()
                    
