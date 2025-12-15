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
st.title("🌎 Forex Sniper Universal (Soporte Multi-Formato)")

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

# Auto-corrección de selección
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

# --- 3. MOTOR DE DATOS (EL TRADUCTOR UNIVERSAL) ---
@st.cache_data(ttl=60) 
def cargar_datos_forex(ticker):
    df_final = pd.DataFrame()
    nombre_limpio = ticker.replace("=X","").replace("=F","") # Ej: JPY, GBPUSD
    
    # Lista inteligente de posibles nombres
    posibles_archivos = [
        f"{nombre_limpio}.csv",              # JPY.csv
        f"USD{nombre_limpio}.csv",           # USDJPY.csv
        f"{nombre_limpio}USD.csv",           # JPYUSD.csv
        f"{nombre_limpio}_PERIOD_H1.csv",
        "USDJPY.csv",                        # Explícito para tu archivo
        "GBPUSD.csv",
        "EURUSD.csv"
    ]
    
    # A. INTENTAR CARGAR CSV HISTÓRICO
    for archivo in posibles_archivos:
        try:
            # Truco 1: Leemos primero asumiendo comas
            df_hist = pd.read_csv(archivo)
            
