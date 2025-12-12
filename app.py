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

# --- FUNCIÓN MAESTRA DE CARGA DE DATOS ---
@st.cache_data
def cargar_datos_hibridos():
    status = st.empty()
    status.text("📂 Cargando histórico 2005-2020 desde CSV...")
    
    # 1. CARGAR DATOS HISTÓRICOS (TU ARCHIVO)
    try:
        # Leemos el archivo asumiendo que está en la misma carpeta en GitHub
        df_hist = pd.read_csv("eurusd_hour.csv")
        
        # Unir Fecha y Hora en una sola columna
        df_hist['Datetime'] = pd.to_datetime(df_hist['Date'] + ' ' + df_hist['Time'])
        df_hist.set_index('Datetime', inplace=True)
        
        # Seleccionamos solo columnas BID y renombramos a estándar inglés
        # BO=Open, BH=High, BL=Low, BC=Close
        df_hist = df_hist[['BO', 'BH', 'BL', 'BC']].rename(
            columns={'BO': 'Open', 'BH': 'High', 'BL': 'Low', 'BC': 'Close'}
        )
        st.toast(f"✅ Histórico cargado: {len(df_hist)} velas (2005-2020)", icon="💾")
        
    except Exception as e:
        st.error(f"No se encontró 'eurusd_hour.csv'. Asegúrate de subirlo a GitHub. Error: {e}")
        return pd.DataFrame()

    # 2. CARGAR DATOS RECIENTES (YAHOO FINANCE)
    status.text("📡 Descargando datos recientes de Yahoo Finance...")
    try:
        # Descargamos máximo permitido por Yahoo para 1h (730 días)
        df_yahoo = yf.download("EURUSD=X", period="2y", interval="1h")
        
        # Limpieza de Yahoo (MultiIndex y Timezone)
        if isinstance(df_yahoo.columns, pd.MultiIndex):
            df_yahoo.columns = df_yahoo.columns.get_level_values(0)
        df_yahoo.index = pd.to_datetime(df_yahoo.index).tz_localize(None)
        
        st.toast(f"✅ Datos recientes cargados: {len(df_yahoo)} velas", icon="📡")
        
    except Exception as e:
        st.error(f"Error con Yahoo Finance: {e}")
        df_yahoo = pd.DataFrame()

    # 3. FUSIÓN DE DATOS
    status.text("🔄 Fusionando datasets...")
    
    # Concatenar y ordenar
    df_final = pd.concat([df_hist, df_yahoo])
    df_final.sort_index(inplace=True)
    
    # Eliminar duplicados (si las fechas se solapan)
    df_final = df_final[~df_final.index.duplicated(keep='last')]
    
    status.empty() # Limpiar mensaje
    return df_final

# --- PROCESAMIENTO TÉCNICO (INDICADORES) ---
def procesar_indicadores(df):
    if df.empty: return df
    
    # Indicadores Técnicos
    df['RSI'] = df.ta.rsi(length=14)
    df['EMA_Fast'] = df.ta.ema(length=50) # Ajustado para 1H
    df['EMA_Slow'] = df.ta.ema(length=200)
    
    # Target: Predecir la SIGUIENTE hora
    # 1 si el cierre de la próxima hora es mayor al actual
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.dropna(inplace=True)
    return df

# --- EJECUCIÓN PRINCIPAL ---
df_raw = cargar_datos_hibridos()

if not df_raw.empty:
    df = procesar_indicadores(df_raw)
    
    # --- VISUALIZACIÓN ---
    st.subheader(f"Análisis Técnico (Total Velas: {len(df):,})")
    
    # Mostrar solo los últimos 7 días para que el gráfico sea legible
    # 7 días * 24 horas = 168 velas
    df_visual = df.tail(168)
    
    fig = go.Figure(data=[go.Candlestick(x=df_visual.index,
                    open=df_visual['Open'],
                    high=df_visual['High'],
                    low=df_visual['Low'],
                    close=df_visual['Close'],
                    name="EUR/USD")])
    
    fig.add_trace(go.Scatter(x=df_visual.index, y=df_visual['EMA_Fast'], line=dict(color='orange', width=1), name="EMA 50"))
    fig.add_trace(go.Scatter(x=df_visual.index, y=df_visual['EMA_Slow'], line=dict(color='blue', width=1), name="EMA 200"))
    
    fig.update_layout(title="Última semana de movimiento (1 Hora)", height=400, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- BOTÓN DE PREDICCIÓN ---
    if st.button('🧠 Entrenar con Histórico Completo y Predecir'):
        
        with st.spinner('Entrenando Inteligencia Artificial... (Esto puede tardar unos segundos)'):
            features = ['RSI', 'EMA_Fast', 'EMA_Slow', 'Open', 'Close', 'High', 'Low']
            
            # Usamos TODOS los datos menos las últimas 500 horas para entrenar
            train = df.iloc[:-500]
            test = df.iloc[-500:]
            
            # Modelo
            model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=42)
            model.fit(train[features], train["Target"])
            
            # Evaluación
            preds = model.predict(test[features])
            precision = precision_score(test["Target"], preds)
            
            # Predicción Futura
            ultimo_dato = df.iloc[-1:][features]
            prediccion_futura = model.predict(ultimo_dato)
            
            # --- MOSTRAR RESULTADOS ---
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Precisión del Modelo", f"{precision:.2%}")
                st.caption("Basado en pruebas con las últimas 500 horas")
            
            with col2:
                direccion = "SUBIRÁ 📈" if prediccion_futura[0] == 1 else "BAJARÁ 📉"
                st.metric("Próxima Hora", direccion)
                
            st.success(f"Modelo entrenado con {len(train):,} velas históricas.")

else:
    st.warning("Esperando datos... Por favor sube 'eurusd_hour.csv' a tu repositorio.")
            
