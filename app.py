import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from datetime import datetime
import pytz

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Forex AI Pro", layout="wide")
st.title("💶 Predicción EUR/USD (Híbrida: Histórico + Yahoo)")

# --- FUNCIÓN DE CÁLCULO CON AUTO-REFRESH (TTL) ---
# ttl=900 significa: "Borra la memoria y descarga nuevo cada 900 segundos (15 min)"
@st.cache_data(ttl=900)
def cargar_datos_v3():
    # 1. CARGAR DATOS HISTÓRICOS (CSV)
    df_hist = pd.DataFrame()
    try:
        df_hist = pd.read_csv("eurusd_hour.csv")
        df_hist['Datetime'] = pd.to_datetime(df_hist['Date'] + ' ' + df_hist['Time'])
        df_hist.set_index('Datetime', inplace=True)
        df_hist = df_hist[['BO', 'BH', 'BL', 'BC']].rename(
            columns={'BO': 'Open', 'BH': 'High', 'BL': 'Low', 'BC': 'Close'}
        )
    except Exception:
        pass 

    # 2. CARGAR DATOS RECIENTES (YAHOO)
    try:
        # Descargamos datos recientes
        df_yahoo = yf.download("EURUSD=X", period="2y", interval="1h")
        
        if isinstance(df_yahoo.columns, pd.MultiIndex):
            df_yahoo.columns = df_yahoo.columns.get_level_values(0)
        
        df_yahoo.index = pd.to_datetime(df_yahoo.index).tz_localize(None)
        
    except Exception:
        df_yahoo = pd.DataFrame()

    # 3. FUSIÓN
    if not df_hist.empty and not df_yahoo.empty:
        df_final = pd.concat([df_hist, df_yahoo])
        df_final = df_final[~df_final.index.duplicated(keep='last')]
    elif not df_hist.empty:
        df_final = df_hist
    else:
        df_final = df_yahoo
        
    df_final.sort_index(inplace=True)
    return df_final

# --- PROCESAMIENTO TÉCNICO ---
def procesar_indicadores(df):
    if df.empty: return df
    
    # Indicadores
    df['RSI'] = df.ta.rsi(length=14)
    df['EMA_Fast'] = df.ta.ema(length=50) 
    df['EMA_Slow'] = df.ta.ema(length=200)
    
    # Target
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.dropna(inplace=True)
    return df

# --- EJECUCIÓN VISUAL ---

# Botón manual para forzar actualización si el usuario quiere
if st.sidebar.button("🔄 Forzar Actualización de Datos"):
    st.cache_data.clear() # Borra la caché manualmente

with st.spinner('Actualizando mercado...'):
    df_raw = cargar_datos_v3()

if not df_raw.empty:
    # Mostrar hora de actualización para seguridad del usuario
    hora_actual = datetime.now(pytz.timezone('America/La_Paz')).strftime("%H:%M:%S")
    st.caption(f"🕒 Última comprobación de datos: {hora_actual} (Hora Bolivia/Local)")
    
    df = procesar_indicadores(df_raw)
    
    # --- GRÁFICO ---
    st.subheader(f"Análisis Técnico (Último precio: {df['Close'].iloc[-1]:.5f})")
    
    df_visual = df.tail(100) # Últimas 100 horas
    
    fig = go.Figure(data=[go.Candlestick(x=df_visual.index,
                    open=df_visual['Open'], high=df_visual['High'],
                    low=df_visual['Low'], close=df_visual['Close'],
                    name="EUR/USD")])
    
    fig.add_trace(go.Scatter(x=df_visual.index, y=df_visual['EMA_Fast'], line=dict(color='orange', width=1), name="EMA 50"))
    fig.add_trace(go.Scatter(x=df_visual.index, y=df_visual['EMA_Slow'], line=dict(color='blue', width=1), name="EMA 200"))
    
    fig.update_layout(height=450, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- PREDICCIÓN ---
    st.write("---")
    col_btn, col_info = st.columns([1, 2])
    
    with col_btn:
        btn_predict = st.button('🧠 Analizar y Predecir', type="primary")
        
    if btn_predict:
        with st.spinner('La IA está pensando...'):
            features = ['RSI', 'EMA_Fast', 'EMA_Slow', 'Open', 'Close', 'High', 'Low']
            
            # Entrenamiento robusto
            train = df.iloc[:-500]
            test = df.iloc[-500:]
            
            model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=42)
            model.fit(train[features], train["Target"])
            
            preds = model.predict(test[features])
            precision = precision_score(test["Target"], preds)
            
            ultimo_dato = df.iloc[-1:][features]
            prediccion_futura = model.predict(ultimo_dato)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Precisión (Backtest)", f"{precision:.2%}")
            with col2:
                direccion = "SUBIRÁ 📈" if prediccion_futura[0] == 1 else "BAJARÁ 📉"
                # Calculamos a qué hora aplica la predicción
                hora_prediccion = df.index[-1] + pd.Timedelta(hours=1)
                st.metric(f"Próximo Cierre (aprox {hora_prediccion.strftime('%H:%M')})", direccion)

else:
    st.error("No se pudieron cargar datos. Verifica tu conexión o el archivo CSV.")
    
