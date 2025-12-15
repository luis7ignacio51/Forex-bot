import streamlit as st
import yfinance as yf
import ccxt  # <--- LIBRERÍA PROFESIONAL DE CRIPTO
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from datetime import datetime, timedelta
import pytz
import time
import numpy as np

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Turbo AI Scalper", layout="wide", page_icon="🚀")
st.title("🚀 Turbo AI Scalper (CCXT + Optimized Feed)")

tz_bolivia = pytz.timezone('America/La_Paz')

# --- 1. CONFIGURACIÓN DE ACTIVOS (MAPPING) ---
# Mapeamos el nombre común al símbolo correcto de cada proveedor
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
st.sidebar.header("🎛️ Panel de Control Turbo")

if 'activo_turbo' not in st.session_state: st.session_state.activo_turbo = "Solana (SOL)"
def actualizar(): st.session_state.activo_turbo = st.session_state.sel_turbo

seleccion = st.sidebar.selectbox("Activo:", list(activos_config.keys()), 
                                 index=list(activos_config.keys()).index(st.session_state.activo_turbo), 
                                 key="sel_turbo", on_change=actualizar)

datos_activo = activos_config[seleccion]
intervalo = st.sidebar.select_slider("Temporalidad:", options=["1m", "5m", "15m", "1h"], value="1h")
umbral = st.sidebar.slider("Confianza IA (%)", 60, 95, 75)
vigilancia = st.sidebar.checkbox("🚨 Auto-Escaneo (Live)", value=False)

# --- 2. MOTOR DE DATOS HÍBRIDO (EL SECRETO) ---
@st.cache_data(ttl=10 if intervalo == "1m" else 30)
def obtener_datos_turbo(config, interval):
    df_final = pd.DataFrame()
    
    # --- A. MOTOR CRIPTO (CCXT - KRAKEN) ---
    # Usamos Kraken porque es amigable con servidores de USA (Streamlit/Colab)
    if config['tipo'] == 'crypto':
        try:
            exchange = ccxt.kraken() 
            timeframe = interval
            # Descargamos velas recientes (Rápido)
            ohlcv = exchange.fetch_ohlcv(config['ticker_ccxt'], timeframe, limit=100)
            
            # Convertir a DataFrame
            df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df.set_index('Datetime', inplace=True)
            
            # Ajustar zona horaria si es necesario (CCXT devuelve UTC)
            df.index = df.index.tz_localize('UTC').tz_convert(None) # Lo dejamos sin zona para compatibilidad
            
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            return df
        except Exception as e:
            st.toast(f"Error CCXT: {e}. Usando Backup Yahoo...", icon="⚠️")
            # Si falla CCXT, pasamos al bloque de Yahoo como respaldo
            pass

    # --- B. MOTOR FOREX / BACKUP (YAHOO OPTIMIZADO) ---
    # Optimizamos Yahoo pidiendo SOLO lo necesario
    try:
        # Mapeo de intervalos para Yahoo
        y_interval = interval
        
        # Periodo ajustado para ser ultra-rápido (menos datos = más velocidad)
        y_period = "1d" if interval == "1m" else ("5d" if interval == "5m" else "1mo")
        
        df = yf.download(config['ticker_y'], period=y_period, interval=y_interval, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        # Limpieza de velas fantasmas de Yahoo (High == Low en M1)
        df = df[df['High'] != df['Low']]
        
        return df
    except Exception as e:
        return pd.DataFrame()

# --- 3. CARGA HISTÓRICA (FUSIÓN) ---
# Fusiona los datos rápidos (Live) con tus CSVs históricos (Learning)
def fusionar_datos(df_live, config):
    if df_live.empty: return df_live
    
    # Intentamos cargar CSV solo si estamos en H1 (para no mezclar peras con manzanas)
    # O si quieres aprender patrones generales
    nombre_csv = config['ticker_y'].replace("=X","").replace("=F","") + ".csv"
    try:
        # Carga simplificada de CSV
        # Nota: Para M1/M5 priorizamos los datos Live, el CSV H1 solo sirve de contexto lejano
        # Por ahora, para Scalping M1, usamos SOLO datos Live para evitar errores de escala
        pass 
    except:
        pass
        
    return df_live

# --- 4. PROCESAMIENTO TÉCNICO ---
def procesar_indicadores(df, interval):
    if len(df) < 20: return pd.DataFrame()
    
    # Indicadores Rápidos para Scalping
    df['EMA_Fast'] = df.ta.ema(length=7)
    df['EMA_Slow'] = df.ta.ema(length=25)
    
    try:
        df['EMA_Trend'] = df.ta.ema(length=100) # Tendencia
    except:
        df['EMA_Trend'] = df['EMA_Slow']
        
    df['RSI'] = df.ta.rsi(length=14)
    df['CCI'] = df.ta.cci(length=14)
    
    # Bandas Bollinger
    bb = df.ta.bbands(length=20, std=2)
    if bb is not None:
        df['BB_Up'] = bb.iloc[:, 1]
        df['BB_Low'] = bb.iloc[:, 0]
        
    # Patrones Matemáticos
    cuerpo = abs(df['Close'] - df['Open'])
    mecha_sup = df['High'] - np.maximum(df['Close'], df['Open'])
    mecha_inf = np.minimum(df['Close'], df['Open']) - df['Low']
    
    # Detectar Pinbars (Martillos/Estrellas)
    df['Pinbar'] = np.where((mecha_inf > cuerpo*2) | (mecha_sup > cuerpo*2), 1, 0)
    
    # Target: 1 si la próxima cierra alcista
    df['Target'] = (df['Close'].shift(-1) > df['Open'].shift(-1)).astype(int)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# --- 5. INTELIGENCIA ARTIFICIAL ---
def consultar_oraculo(df):
    features = ['RSI', 'CCI', 'EMA_Fast', 'EMA_Slow', 'BB_Up', 'BB_Low', 'Pinbar']
    features = [f for f in features if f in df.columns]
    
    if len(df) < 30: return 0, 0.0 # Seguridad
    
    # Entrenamiento Flash (Solo datos recientes para adaptarse al mercado actual)
    train = df.iloc[:-1]
    
    try:
        model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(train[features], train["Target"])
        
        ultimo = df.iloc[-1:][features]
        pred = model.predict(ultimo)[0]
        prob = model.predict_proba(ultimo)[0]
        confianza = max(prob)
        
        # Filtro Anti-Alucinación (Yahoo Fix)
        if confianza > 0.999: return 0, 0.0
        
        return pred, confianza
    except:
        return 0, 0.0

# --- 6. INTERFAZ VISUAL TURBO ---
if st.sidebar.button("⚡ Forzar Recarga"): st.cache_data.clear()
placeholder = st.empty()

with placeholder.container():
    # 1. Obtener Datos
    df_live = obtener_datos_turbo(datos_activo, intervalo)
    
    if not df_live.empty:
        df = procesar_indicadores(df_live, intervalo)
        
        if not df.empty:
            pred, conf = consultar_oraculo(df)
            
            # Variables
            precio = df['Close'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            trend = df['EMA_Trend'].iloc[-1]
            
            # Decisión
            decision = "ESPERAR"
            razon = "Escaneando..."
            color = "#e9ecef"; txt = "#333"; icono = "⏳"
            
            # LÓGICA DE TRADING
            if conf*100 >= umbral:
                if pred == 1: # CALL
                    # Filtro de Tendencia o Rebote
                    if precio > trend or rsi < 30:
                        decision = "CALL (SUBIR) 🚀"
                        razon = "Patrón Alcista Confirmado"
                        color = "#d4edda"; txt = "#155724"; icono = "📈"
                else: # PUT
                    if precio < trend or rsi > 70:
                        decision = "PUT (BAJAR) 📉"
                        razon = "Patrón Bajista Confirmado"
                        color = "#f8d7da"; txt = "#721c24"; icono = "📉"
            
            # Tiempo Restante
            now = datetime.now(tz_bolivia)
            if intervalo == "1m": resto = 60 - now.second
            elif intervalo == "5m": resto = (5 - (now.minute % 5)) * 60 - now.second
            else: resto = (60 - now.minute) * 60 - now.second
            
            # UI
            c1, c2 = st.columns([2,1])
            c1.markdown(f"### {seleccion} [{intervalo}]")
            c1.markdown(f"<h1 style='margin:0'>${precio:.5f}</h1>", unsafe_allow_html=True)
            
            # Badge de Fuente de Datos
            source_badge = "🟢 CONEXIÓN DIRECTA (CCXT)" if datos_activo['tipo'] == 'crypto' else "🟠 CONEXIÓN WEB (YAHOO)"
            c2.caption(source_badge)
            c2.metric("Cierre Vela:", f"{resto}s")
            
            st.markdown(f"""
            <div style="background-color: {color}; padding: 15px; border-radius: 10px; border: 2px solid {txt}; text-align: center;">
                <h1 style="color: {txt}; margin:0;">{icono} {decision}</h1>
                <p style="color: {txt}; margin:0;"><b>{razon}</b></p>
                <p style="color: {txt}; font-size:12px;">Confianza IA: {conf:.1%} | RSI: {rsi:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Gráfico Simple
            fig = go.Figure(data=[go.Candlestick(x=df.tail(50).index, open=df.tail(50)['Open'], high=df.tail(50)['High'], low=df.tail(50)['Low'], close=df.tail(50)['Close'])])
            fig.add_trace(go.Scatter(x=df.tail(50).index, y=df.tail(50)['EMA_Fast'], line=dict(color='orange', width=1)))
            fig.add_trace(go.Scatter(x=df.tail(50).index, y=df.tail(50)['EMA_Trend'], line=dict(color='blue', width=1)))
            fig.update_layout(height=350, xaxis_rangeslider_visible=False, margin=dict(t=10,b=0), template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Recopilando datos para calcular indicadores... Espera unos segundos.")
    else:
        st.error("Error de conexión. Si usas Forex en M1, Yahoo puede estar saturado. Prueba M5 o Cripto.")

if vigilancia:
    time.sleep(5 if intervalo == "1m" else 15)
    st.rerun()
        
