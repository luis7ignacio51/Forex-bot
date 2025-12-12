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

# --- FUNCIÓN DE CÁLCULO (SIN ELEMENTOS VISUALES) ---
@st.cache_data
def obtener_datos_hibridos():
    # 1. CARGAR DATOS HISTÓRICOS (CSV)
    df_hist = pd.DataFrame()
    try:
        # Intentamos leer el archivo. Si no existe, seguirá vacío.
        df_hist = pd.read_csv("eurusd_hour.csv")
        
        # Limpieza del CSV
        df_hist['Datetime'] = pd.to_datetime(df_hist['Date'] + ' ' + df_hist['Time'])
        df_hist.set_index('Datetime', inplace=True)
        
        # Seleccionamos solo columnas BID y renombramos
        df_hist = df_hist[['BO', 'BH', 'BL', 'BC']].rename(
            columns={'BO': 'Open', 'BH': 'High', 'BL': 'Low', 'BC': 'Close'}
        )
    except Exception:
        # Si falla (no está el archivo), simplemente ignoramos esta parte
        pass

    # 2. CARGAR DATOS RECIENTES (YAHOO)
    try:
        df_yahoo = yf.download("EURUSD=X", period="2y", interval="1h")
        
        # Limpieza de Yahoo (MultiIndex y Timezone)
        if isinstance(df_yahoo.columns, pd.MultiIndex):
            df_yahoo.columns = df_yahoo.columns.get_level_values(0)
        
        # Convertir índice a datetime y quitar zona horaria para compatibilidad
        df_yahoo.index = pd.to_datetime(df_yahoo.index).tz_localize(None)
        
    except Exception:
        df_yahoo = pd.DataFrame()

    # 3. FUSIÓN
    if not df_hist.empty and not df_yahoo.empty:
        df_final = pd.concat([df_hist, df_yahoo])
        # Eliminar duplicados priorizando Yahoo (más reciente)
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
    
    # Target: Predecir la SIGUIENTE hora
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.dropna(inplace=True)
    return df

# --- EJECUCIÓN VISUAL (AQUÍ SÍ PODEMOS USAR UI) ---

# Usamos un spinner mientras carga la función caché
with st.spinner('Cargando y fusionando bases de datos...'):
    df_raw = obtener_datos_hibridos()

if not df_raw.empty:
    st.toast(f"Datos cargados: {len(df_raw)} velas totales", icon="✅")
    
    df = procesar_indicadores(df_raw)
    
    # --- VISUALIZACIÓN ---
    st.subheader(f"Análisis Técnico (Total Velas: {len(df):,})")
    
    # Última semana (168 horas)
    df_visual = df.tail(168)
    
    fig = go.Figure(data=[go.Candlestick(x=df_visual.index,
                    open=df_visual['Open'], high=df_visual['High'],
                    low=df_visual['Low'], close=df_visual['Close'],
                    name="EUR/USD")])
    
    fig.add_trace(go.Scatter(x=df_visual.index, y=df_visual['EMA_Fast'], line=dict(color='orange', width=1), name="EMA 50"))
    fig.add_trace(go.Scatter(x=df_visual.index, y=df_visual['EMA_Slow'], line=dict(color='blue', width=1), name="EMA 200"))
    
    fig.update_layout(height=400, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- BOTÓN DE PREDICCIÓN ---
    if st.button('🧠 Entrenar IA y Predecir Siguiente Hora'):
        
        with st.spinner('Entrenando Inteligencia Artificial...'):
            features = ['RSI', 'EMA_Fast', 'EMA_Slow', 'Open', 'Close', 'High', 'Low']
            
            # Entrenamiento (Todo menos las últimas 500 horas)
            train = df.iloc[:-500]
            test = df.iloc[-500:]
            
            model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=42)
            model.fit(train[features], train["Target"])
            
            # Evaluación
            preds = model.predict(test[features])
            precision = precision_score(test["Target"], preds)
            
            # Predicción Futura
            ultimo_dato = df.iloc[-1:][features]
            prediccion_futura = model.predict(ultimo_dato)
            
            # Resultados
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Precisión (Backtest)", f"{precision:.2%}")
            with col2:
                direccion = "SUBIRÁ 📈" if prediccion_futura[0] == 1 else "BAJARÁ 📉"
                st.metric("Próxima Hora", direccion)

else:
    st.error("No hay datos. Asegúrate de haber subido 'eurusd_hour.csv' a GitHub o que Yahoo Finance esté funcionando.")
    
