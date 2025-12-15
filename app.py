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
st.set_page_config(page_title="Scalping AI Master v17", layout="wide", page_icon="🛡️")
st.title("🛡️ Scalping AI Master v17 (Bulletproof)")

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

# 2. Selector de Temporalidad
intervalo = st.sidebar.select_slider("Temporalidad:", options=["1m", "5m", "15m", "1h"], value="1h")

# 3. Ajustes
umbral = st.sidebar.slider("Confianza IA (%)", 55, 95, 70)
vigilancia = st.sidebar.checkbox("🚨 Auto-Escaneo (Live)", value=False)

# --- 1. MOTOR DE DATOS ---
@st.cache_data(ttl=30 if intervalo in ['1m','5m'] else 60) 
def cargar_datos_multitimeframe(ticker, interval):
    df_final = pd.DataFrame()
    nombre_limpio = ticker.replace("=X","").replace("=F","")
    
    usar_csv = False
    if interval == "1h":
        archivos = [f"{nombre_limpio}.csv", "EURUSD.csv", "GBPUSD.csv", "USDJPY.csv", "SOL-USD.csv"]
        for f in archivos:
            try:
                df = pd.read_csv(f, sep=None, engine='python')
                df.columns = df.columns.str.replace('<','').str.replace('>','').str.capitalize()
                
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

    try:
        # Descargamos más datos para asegurar que los indicadores tengan historial
        per = "5d" if interval == "1m" else ("60d" if interval in ["5m","15m"] else "2y")
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
        # Limpieza inicial básica
        df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna()
        
    return df_final

# --- 2. PROCESAMIENTO ---
def procesar_scalping(df, interval):
    if df.empty: return df
    
    # Asegurar que hay suficientes datos
    if len(df) < 50: return pd.DataFrame() # Retorna vacío si no hay datos suficientes
    
    len_rapida = 9 if interval == "1m" else 20
    len_lenta = 21 if interval == "1m" else 50
    
    df['EMA_Rapida'] = df.ta.ema(length=len_rapida)
    df['EMA_Lenta'] = df.ta.ema(length=len_lenta) 
    
    # EMA 200 puede fallar si hay pocos datos (ej. M1 reciente), usamos try/except
    try:
        df['EMA_200'] = df.ta.ema(length=200)
    except:
        df['EMA_200'] = df['EMA_Lenta'] # Fallback
        
    df['RSI'] = df.ta.rsi(length=14)
    
    bb = df.ta.bbands(length=20, std=2)
    if bb is not None:
        df['BB_Up'] = bb.iloc[:, 1]
        df['BB_Low'] = bb.iloc[:, 0]
        
    df['CCI'] = df.ta.cci(length=14)

    cuerpo = abs(df['Close'] - df['Open'])
    mecha_sup = df['High'] - np.maximum(df['Close'], df['Open'])
    mecha_inf = np.minimum(df['Close'], df['Open']) - df['Low']
    
    df['Patron_Bull'] = np.where((mecha_inf > cuerpo*1.5) & (mecha_sup < cuerpo*0.5), 1, 0)
    df['Patron_Bear'] = np.where((mecha_sup > cuerpo*1.5) & (mecha_inf < cuerpo*0.5), 1, 0)

    df['Target'] = (df['Close'].shift(-1) > df['Open'].shift(-1)).astype(int)
    
    # LIMPIEZA CRÍTICA PARA EVITAR ERRORES DE SKLEARN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df

# --- 3. IA (CON PROTECCIÓN ANTI-CRASH) ---
def ejecutar_ia_scalper(df):
    features = ['RSI', 'CCI', 'EMA_Rapida', 'EMA_Lenta', 'BB_Up', 'BB_Low', 'Patron_Bull', 'Patron_Bear']
    
    # Validar que las columnas existen
    features = [f for f in features if f in df.columns]
    
    # Validar que hay datos limpios para entrenar
    if len(df) < 50:
        return 0, 0.0 # No hay suficientes datos seguros
    
    limit = 1000 
    train = df.iloc[-limit:-1] # Entrenar con el pasado
    
    # Doble chequeo de limpieza en el set de entrenamiento
    train = train.dropna()
    
    if len(train) < 10:
        return 0, 0.0 # Error de seguridad
    
    try:
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(train[features], train["Target"])
        
        ultimo = df.iloc[-1:][features]
        # Chequeo final del dato actual
        if ultimo.isnull().values.any():
            return 0, 0.0 # El dato actual está corrupto
            
        pred = model.predict(ultimo)[0]
        prob = model.predict_proba(ultimo)[0]
        return pred, max(prob)
        
    except Exception as e:
        st.error(f"Error interno IA: {e}")
        return 0, 0.0

# --- 4. INTERFAZ ---
if st.sidebar.button("Forzar Recarga"): st.cache_data.clear()
placeholder = st.empty()

with placeholder.container():
    df_raw = cargar_datos_multitimeframe(ticker, intervalo)
    
    if not df_raw.empty:
        df = procesar_scalping(df_raw, intervalo)
        
        if not df.empty:
            pred, conf = ejecutar_ia_scalper(df)
            
            # Datos
            precio = df['Close'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            # Usamos EMA 200 si existe, si no la Lenta
            ema_trend = df['EMA_200'].iloc[-1] if 'EMA_200' in df.columns else df['EMA_Lenta'].iloc[-1]
            
            # --- LÓGICA DE DIAGNÓSTICO ---
            decision = "ESPERAR"
            razon = "Analizando..."
            color_box = "#e9ecef"; txt_color = "#333"; icono = "⏳"
            
            # 1. Chequeo de Confianza IA
            if conf*100 < umbral:
                razon = f"Confianza IA Baja ({conf:.1%} < {umbral}%)"
                icono = "🤖"
            elif conf == 0.0:
                razon = "Datos insuficientes para cálculo seguro"
                icono = "⚠️"
            else:
                # 2. Chequeo de Filtros Técnicos
                if pred == 1: # IA dice SUBIR
                    if df['Close'].iloc[-1] > ema_trend: # A favor de tendencia
                        decision = "CALL (ALZA) 🚀"
                        razon = "IA + Tendencia Alcista Confirmada"
                        color_box = "#d4edda"; txt_color = "#155724"; icono = "📈"
                    elif rsi < 30: # Rebote
                        decision = "CALL (REBOTE) 🚀"
                        razon = "IA + RSI Sobrevendido (Rebote)"
                        color_box = "#d4edda"; txt_color = "#155724"; icono = "📈"
                    else:
                        decision = "ESPERAR ✋"
                        razon = "IA Alcista pero Tendencia Bajista (Riesgo)"
                        color_box = "#fff3cd"; txt_color = "#856404"; icono = "⚠️"
                        
                elif pred == 0: # IA dice BAJAR
                    if df['Close'].iloc[-1] < ema_trend: # A favor de tendencia
                        decision = "PUT (BAJA) 📉"
                        razon = "IA + Tendencia Bajista Confirmada"
                        color_box = "#f8d7da"; txt_color = "#721c24"; icono = "📉"
                    elif rsi > 70: # Rebote
                        decision = "PUT (REBOTE) 📉"
                        razon = "IA + RSI Sobrecomprado (Rebote)"
                        color_box = "#f8d7da"; txt_color = "#721c24"; icono = "📉"
                    else:
                        decision = "ESPERAR ✋"
                        razon = "IA Bajista pero Tendencia Alcista (Riesgo)"
                        color_box = "#fff3cd"; txt_color = "#856404"; icono = "⚠️"
            
            # Tiempo
            now = datetime.now(tz_bolivia)
            minutos_actuales = now.minute
            
            if intervalo == "1m": 
                resto = 60 - now.second
                delta_cierre = timedelta(seconds=resto)
            elif intervalo == "5m": 
                resto = (5 - (minutos_actuales % 5)) * 60 - now.second
                delta_cierre = timedelta(seconds=resto)
            elif intervalo == "15m": 
                resto = (15 - (minutos_actuales % 15)) * 60 - now.second
                delta_cierre = timedelta(seconds=resto)
            else: 
                resto = (60 - minutes_actuales) * 60 - now.second if 'minutes_actuales' in locals() else (60 - minutos_actuales) * 60 - now.second
                delta_cierre = timedelta(seconds=resto)
            
            hora_cierre_vela = (now + delta_cierre).strftime("%H:%M:%S")

            # --- UI MEJORADA ---
            c1, c2 = st.columns([2,1])
            c1.markdown(f"### {seleccion} [{intervalo}]")
            c1.markdown(f"<h1 style='margin:0'>${precio:.5f}</h1>", unsafe_allow_html=True)
            
            c2.metric("Cierra en:", f"{resto} seg")
            c2.caption(f"Predicción para vela que cierra a las: {hora_cierre_vela}")
            
            st.markdown(f"""
            <div style="background-color: {color_box}; padding: 20px; border-radius: 15px; border: 3px solid {txt_color}; text-align: center;">
                <h1 style="color: {txt_color}; margin:0; font-size: 40px;">{icono}</h1>
                <h2 style="color: {txt_color}; margin:0;">{decision}</h2>
                <hr style="border-color: {txt_color}; opacity: 0.2">
                <p style="color: {txt_color}; margin:0; font-weight: bold;">Diagnóstico: {razon}</p>
                <p style="color: {txt_color}; margin:0; font-size: 14px;">Confianza IA: {conf:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            df_ver = df.tail(60) 
            fig = go.Figure(data=[go.Candlestick(x=df_ver.index, open=df_ver['Open'], high=df_ver['High'], low=df_ver['Low'], close=df_ver['Close'])])
            fig.add_trace(go.Scatter(x=df_ver.index, y=df_ver['EMA_Rapida'], line=dict(color='orange', width=1), name='EMA Rápida'))
            fig.add_trace(go.Scatter(x=df_ver.index, y=df_ver['EMA_Lenta'], line=dict(color='cyan', width=1), name='EMA Lenta'))
            fig.add_trace(go.Scatter(x=df_ver.index, y=df_ver['BB_Up'], line=dict(color='gray', width=1, dash='dot'), name='BB Sup'))
            fig.add_trace(go.Scatter(x=df_ver.index, y=df_ver['BB_Low'], line=dict(color='gray', width=1, dash='dot'), name='BB Inf'))
            fig.update_layout(height=400, template="plotly_white", xaxis_rangeslider_visible=False, margin=dict(t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Datos insuficientes para generar indicadores (NaN error). Intenta una temporalidad mayor o espera a que el mercado abra.")
            
    else:
        st.warning("Cargando datos... (Recuerda: M1 solo tiene datos recientes de Yahoo)")

if vigilancia:
    sleep_time = 10 if intervalo == "1m" else (30 if intervalo == "5m" else 60)
    time.sleep(sleep_time)
    st.rerun()
                
