import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from datetime import datetime
import pytz

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Forex AI Quant", layout="wide")
st.title("💶 Predicción EUR/USD (Nivel Quant v4.0)")

# --- 1. FUNCIÓN DE CARGA DE DATOS (AUTO-ACTUALIZABLE) ---
# ttl=900 (15 min) para refrescar datos automáticamente
# Cambiamos nombre a 'v4' para limpiar cualquier error de caché previo
@st.cache_data(ttl=900)
def cargar_datos_v4():
    # A. Intentar cargar histórico CSV
    df_hist = pd.DataFrame()
    try:
        df_hist = pd.read_csv("eurusd_hour.csv")
        # Crear índice de fecha
        df_hist['Datetime'] = pd.to_datetime(df_hist['Date'] + ' ' + df_hist['Time'])
        df_hist.set_index('Datetime', inplace=True)
        # Estandarizar nombres
        df_hist = df_hist[['BO', 'BH', 'BL', 'BC']].rename(
            columns={'BO': 'Open', 'BH': 'High', 'BL': 'Low', 'BC': 'Close'}
        )
    except Exception:
        pass # Si no hay CSV, seguimos

    # B. Descargar datos recientes Yahoo Finance
    try:
        df_yahoo = yf.download("EURUSD=X", period="2y", interval="1h")
        
        # Aplanar columnas si es necesario
        if isinstance(df_yahoo.columns, pd.MultiIndex):
            df_yahoo.columns = df_yahoo.columns.get_level_values(0)
        
        # Quitar zona horaria para compatibilidad
        df_yahoo.index = pd.to_datetime(df_yahoo.index).tz_localize(None)
    except Exception:
        df_yahoo = pd.DataFrame()

    # C. Fusionar
    if not df_hist.empty and not df_yahoo.empty:
        df_final = pd.concat([df_hist, df_yahoo])
        # Eliminar duplicados (prioridad al más reciente)
        df_final = df_final[~df_final.index.duplicated(keep='last')]
    elif not df_hist.empty:
        df_final = df_hist
    else:
        df_final = df_yahoo
        
    df_final.sort_index(inplace=True)
    return df_final

# --- 2. PROCESAMIENTO E INDICADORES AVANZADOS ---
def procesar_indicadores_avanzados(df):
    if df.empty: return df
    
    # --- Tendencia ---
    df['EMA_Fast'] = df.ta.ema(length=50)
    df['EMA_Slow'] = df.ta.ema(length=200)
    
    # MACD (Devuelve 3 columnas: MACD, Histograma, Señal)
    # Usaremos los nombres estándar que genera pandas_ta
    df.ta.macd(append=True) 
    
    # --- Fuerza de Tendencia ---
    # ADX: Si es bajo (<20), el mercado está lateral (peligroso operar)
    df.ta.adx(append=True)
    
    # --- Volatilidad ---
    # ATR: Rango promedio verdadero
    df.ta.atr(append=True)
    
    # --- Momentum ---
    df['RSI'] = df.ta.rsi(length=14)

    # --- Contexto Temporal (NUEVO) ---
    # La IA aprenderá patrones horarios (ej: volatilidad de Londres vs Asia)
    df['Hora'] = df.index.hour
    df['DiaSemana'] = df.index.dayofweek # 0=Lunes, 4=Viernes

    # --- Target (Objetivo) ---
    # 1 si el cierre de la PRÓXIMA hora es mayor al actual
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Limpieza de nulos generados por indicadores
    df.dropna(inplace=True)
    return df

# --- 3. INTERFAZ DE USUARIO ---

# Botón lateral para forzar recarga
if st.sidebar.button("🔄 Forzar Recarga de Datos"):
    st.cache_data.clear()

# Carga de datos
with st.spinner('Analizando mercado con indicadores avanzados...'):
    df_raw = cargar_datos_v4()

if not df_raw.empty:
    # Procesar
    df = procesar_indicadores_avanzados(df_raw)
    
    # Info de estado
    hora_bolivia = datetime.now(pytz.timezone('America/La_Paz')).strftime("%H:%M")
    st.caption(f"📅 Datos actualizados. Hora local: {hora_bolivia} | Velas analizadas: {len(df):,}")
    
    # --- GRÁFICO PRINCIPAL ---
    st.subheader(f"Precio Actual: {df['Close'].iloc[-1]:.5f}")
    
    # Últimas 100 horas para visualización limpia
    df_visual = df.tail(100)
    
    fig = go.Figure(data=[go.Candlestick(x=df_visual.index,
                    open=df_visual['Open'], high=df_visual['High'],
                    low=df_visual['Low'], close=df_visual['Close'],
                    name="EUR/USD")])
    
    # Añadimos EMAs
    fig.add_trace(go.Scatter(x=df_visual.index, y=df_visual['EMA_Fast'], line=dict(color='orange', width=1), name="EMA 50"))
    fig.add_trace(go.Scatter(x=df_visual.index, y=df_visual['EMA_Slow'], line=dict(color='blue', width=1), name="EMA 200"))
    
    fig.update_layout(height=400, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # --- SECCIÓN DE INTELIGENCIA ARTIFICIAL ---
    st.markdown("### 🧠 Cerebro Digital")
    
    if st.button('Ejecutar Análisis Predictivo', type="primary"):
        
        with st.spinner('Entrenando modelos y evaluando variables...'):
            # Definimos las columnas exactas que usará la IA
            # Nota: pandas_ta genera nombres específicos como 'MACD_12_26_9'
            features = [
                'RSI', 'EMA_Fast', 'EMA_Slow', 'Open', 'Close', 
                'MACD_12_26_9', 'MACDh_12_26_9', # MACD y su Histograma
                'ADX_14',   # Fuerza de tendencia
                'ATRr_14',  # Volatilidad
                'Hora', 'DiaSemana' # Contexto tiempo
            ]
            
            # Verificar que las columnas existan (por si acaso cambian nombres)
            features_existentes = [col for col in features if col in df.columns]
            
            # Separar datos (Entrenamiento vs Test)
            train = df.iloc[:-500] # Todo menos lo último
            test = df.iloc[-500:]  # Últimas 500 horas para probar
            
            # Crear y entrenar modelo
            model = RandomForestClassifier(n_estimators=150, min_samples_split=50, random_state=42)
            model.fit(train[features_existentes], train["Target"])
            
            # Evaluar precisión
            preds = model.predict(test[features_existentes])
            precision = precision_score(test["Target"], preds)
            
            # Predecir futuro inmediato
            ultimo_dato = df.iloc[-1:][features_existentes]
            prediccion = model.predict(ultimo_dato)
            probabilidad = model.predict_proba(ultimo_dato)[0] # Confianza de la IA
            
            # --- RESULTADOS ---
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Precisión Histórica", f"{precision:.2%}")
                st.caption("Efectividad en las últimas 500h")
                
            with col2:
                # Decisión
                direccion = "ALCISTA (SUBIR) 📈" if prediccion[0] == 1 else "BAJISTA (BAJAR) 📉"
                # Color del texto según dirección
                color = "green" if prediccion[0] == 1 else "red"
                st.markdown(f"Predicción:<br><span style='color:{color}; font-size:24px; font-weight:bold'>{direccion}</span>", unsafe_allow_html=True)
                
            with col3:
                # Confianza del modelo
                confianza = max(probabilidad)
                st.metric("Confianza de la IA", f"{confianza:.1%}")
                if confianza < 0.55:
                    st.warning("⚠️ Confianza baja. Precaución.")

            # --- EXPLICABILIDAD (¿POR QUÉ DICE ESO?) ---
            st.write("---")
            st.subheader("¿Qué está mirando la IA?")
            
            importances = pd.DataFrame({
                'Indicador': features_existentes,
                'Importancia': model.feature_importances_
            }).sort_values(by='Importancia', ascending=True)
            
            # Gráfico de barras
            fig_imp = go.Figure(go.Bar(
                x=importances['Importancia'],
                y=importances['Indicador'],
                orientation='h',
                marker=dict(color='purple')
            ))
            fig_imp.update_layout(title="Peso de cada indicador en la decisión", height=350)
            st.plotly_chart(fig_imp, use_container_width=True)
            
            st.info(f"Nota: La predicción aplica para el cierre de la vela de las {(df.index[-1] + pd.Timedelta(hours=1)).strftime('%H:%M')}")

else:
    st.error("Error cargando datos. Verifica que 'eurusd_hour.csv' esté en GitHub.")
    
