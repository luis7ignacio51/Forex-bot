import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Forex AI Pro", layout="wide")
st.title("💶 Predicción EUR/USD (Híbrida: Histórico + Yahoo)")

# --- FUNCIÓN DE CÁLCULO (RENOMBRADA PARA LIMPIAR CACHÉ) ---
# Al cambiar el nombre a 'cargar_datos_v2', forzamos a Streamlit a olvidar el error anterior.
@st.cache_data
def cargar_datos_v2():
    # 1. CARGAR DATOS HISTÓRICOS (CSV)
    df_hist = pd.DataFrame()
    try:
        # Intentamos leer el archivo.
        df_hist = pd.read_csv("eurusd_hour.csv")
        
        # Limpieza del CSV
        df_hist['Datetime'] = pd.to_datetime(df_hist['Date'] + ' ' + df_hist['Time'])
        df_hist.set_index('Datetime', inplace=True)
        
        # Seleccionamos columnas y renombramos
        df_hist = df_hist[['BO', 'BH', 'BL', 'BC']].rename(
            columns={'BO': 'Open', 'BH': 'High', 'BL': 'Low', 'BC': 'Close'}
        )
    except Exception:
        pass # Si falla, seguimos sin histórico

    # 2. CARGAR DATOS RECIENTES (YAHOO)
    try:
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

with st.spinner('Cargando datos (v2)...'):
    # Llamamos a la nueva función
    df_raw = cargar_datos_v2()

if not df_raw.empty:
    st.toast(f"Datos cargados: {len(df_raw)} velas", icon="✅")
    
    df = procesar_indicadores(df_raw)
    
    # --- GRÁFICO ---
    st.subheader(f"Análisis Técnico (Total: {len(df):,})")
    
    df_visual = df.tail(168) # Última semana
    
    fig = go.Figure(data=[go.Candlestick(x=df_visual.index,
                    open=df_visual['Open'], high=df_visual['High'],
                    low=df_visual['Low'], close=df_visual['Close'],
                    name="EUR/USD")])
    
    fig.add_trace(go.Scatter(x=df_visual.index, y=df_visual['EMA_Fast'], line=dict(color='orange', width=1), name="EMA 50"))
    fig.add_trace(go.Scatter(x=df_visual.index, y=df_visual['EMA_Slow'], line=dict(color='blue', width=1), name="EMA 200"))
    
    fig.update_layout(height=400, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- BOTÓN DE PREDICCIÓN ---
    if st.button('🧠 Entrenar IA y Predecir'):
        
        with st.spinner('Entrenando...'):
            features = ['RSI', 'EMA_Fast', 'EMA_Slow', 'Open', 'Close', 'High', 'Low']
            
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
                st.metric("Precisión Histórica", f"{precision:.2%}")
            with col2:
                direccion = "SUBIRÁ 📈" if prediccion_futura[0] == 1 else "BAJARÁ 📉"
                st.metric("Próxima Hora", direccion)

else:
    st.error("No se pudieron cargar datos. Verifica que 'eurusd_hour.csv' esté en GitHub.")
