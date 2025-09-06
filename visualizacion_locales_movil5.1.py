# -*- coding: utf-8 -*-
"""
visualizacion_locales.py — Mobile-optimized with fixed selectbox for long activity names
"""

import os
import json
import unicodedata
import base64, mimetypes
from pathlib import Path
from textwrap import dedent
import requests

import pandas as pd
import numpy as np
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from branca.colormap import linear

# =========================
# CONFIG
# =========================
CSV_FILE = "actividadeconomica_enriquecido.csv"
SEP = ";"  # Your data uses semicolon separator
CACHE_FILE = "geocode_cache.csv"
CITY_DEFAULT = "Madrid"
COUNTRY_DEFAULT = "España"

# Google Drive file ID - Your actual file ID
GOOGLE_DRIVE_FILE_ID = "14HcbItRNMbSxCbim5s0r9J7FJ6SrN7zW"  # Madrid business opportunities CSV

# GeoJSONs (en assets/)
BARRIOS_GEOJSON_PATH = os.path.join("assets", "barrios_madrid.geojson")
DISTRITOS_GEOJSON_PATH = os.path.join("assets", "distritos_madrid.geojson")
BARRIO_PROP_KEY = "NOMBRE"
DISTRITO_PROP_KEY = "NOMBRE"

# =========================
# GOOGLE DRIVE DOWNLOAD FUNCTION
# =========================
@st.cache_data(show_spinner=True)
def download_csv_from_drive(file_id, destination):
    """Simple function to download CSV if it doesn't exist"""
    if not os.path.exists(CSV_FILE):
        st.info("📥 Dataset not found locally. Attempting download from Google Drive...")
        
        # Try multiple download URLs
        urls_to_try = [
            f"https://drive.usercontent.google.com/download?id={GOOGLE_DRIVE_FILE_ID}&export=download&confirm=t",
            f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}&export=download&confirm=t",
            f"https://docs.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}&export=download"
        ]
        
        success = False
        for i, url in enumerate(urls_to_try):
            try:
                st.write(f"Trying download method {i+1}...")
                response = requests.get(url, timeout=30)
                
                # Check if we got actual CSV data (not HTML error page)
                if response.status_code == 200 and 'text/html' not in response.headers.get('content-type', ''):
                    with open(CSV_FILE, 'wb') as f:
                        f.write(response.content)
                    
                    # Verify file size
                    if os.path.getsize(CSV_FILE) > 1000000:  # At least 1MB
                        st.success("✅ Dataset downloaded successfully!")
                        success = True
                        break
                    else:
                        os.remove(CSV_FILE)  # Remove tiny file
                        
            except Exception as e:
                st.write(f"Method {i+1} failed: {str(e)}")
                continue
        
        if not success:
            st.error("❌ Could not download the dataset automatically.")
            st.markdown(f"""
            **Manual download required:**
            1. [Click here to download the CSV file](https://drive.google.com/file/d/{GOOGLE_DRIVE_FILE_ID}/view?usp=sharing)
            2. Save it as `{CSV_FILE}` in your project folder
            3. Reload this page
            """)
            st.stop()


@st.cache_data
def ensure_dataset_available():
    """Ensure the dataset is available locally, download if necessary"""
    if not os.path.exists(CSV_FILE):
        st.info("📥 Dataset not found locally. Downloading from Google Drive...")
        download_csv_from_drive(GOOGLE_DRIVE_FILE_ID, CSV_FILE)
    
    # Verify file exists and is not empty
    if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
        return CSV_FILE
    else:
        st.error("❌ Dataset file is missing or empty!")
        st.stop()

# =========================
# UTILIDADES
# =========================
def normalize_str(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    return s.upper()

def build_address(row):
    parts = []
    via = normalize_str(row.get("desc_vial_edificio"))
    clase = normalize_str(row.get("clase_vial_edificio"))
    numero = str(row.get("num_edificio")) if not pd.isna(row.get("num_edificio")) else ""
    barrio = normalize_str(row.get("desc_barrio_local"))
    distrito = normalize_str(row.get("desc_distrito_local"))

    if via:
        parts.append(f"{clase.title()} {via.title()}" if clase and clase not in via else via.title())
    if numero and numero not in ("nan", "None"):
        parts.append(str(int(float(numero))) if numero.replace(".", "", 1).isdigit() else numero)
    if distrito: parts.append(distrito.title())
    if barrio and barrio not in distrito: parts.append(barrio.title())
    parts.append(CITY_DEFAULT)
    parts.append(COUNTRY_DEFAULT)
    parts = [p for p in parts if p and p not in ("NAN", "NONE")]
    return ", ".join(parts)

def load_geojson(path, prop_key):
    if not path or not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        geo = json.load(f)
    name_to_feature = {}
    for feat in geo.get("features", []):
        props = feat.get("properties", {})
        name = props.get(prop_key) or props.get(prop_key.lower()) or (list(props.values())[0] if props else None)
        if name is None:
            continue
        name_to_feature[normalize_str(name)] = feat
    return geo, name_to_feature

def quantile_labels(s, q=[0, .25, .5, .75, 1.0], labels=("Muy baja", "Baja", "Media", "Alta")):
    if s.nunique() <= 1:
        return pd.Series(["Sin datos"] * len(s), index=s.index)
    
    # Handle cases with very few unique values
    unique_vals = s.nunique()
    
    try:
        return pd.qcut(s, q=q, duplicates="drop", labels=labels[:(len(q)-1)])
    except ValueError:
        # If qcut fails, use a simpler binning approach
        if unique_vals == 2:
            # Only two unique values: use simple binary classification
            median_val = s.median()
            return s.apply(lambda x: "Baja" if x <= median_val else "Alta")
        elif unique_vals == 3:
            # Three unique values: use tertiles
            q33, q67 = s.quantile([0.33, 0.67])
            return s.apply(lambda x: "Muy baja" if x <= q33 else ("Baja" if x <= q67 else "Media"))
        else:
            # For other cases, try with fewer bins
            try:
                return pd.qcut(s, q=min(unique_vals, 3), duplicates="drop", 
                             labels=["Baja", "Media", "Alta"][:min(unique_vals-1, 3)])
            except ValueError:
                # Final fallback: simple binary classification
                median_val = s.median()
                return s.apply(lambda x: "Baja" if x <= median_val else "Alta")

def feature_centroid(feat):
    try:
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            coords = np.array(geom["coordinates"][0])
        elif geom["type"] == "MultiPolygon":
            coords = np.array(geom["coordinates"][0][0])
        else:
            return np.nan, np.nan
        lon = coords[:, 0].mean()
        lat = coords[:, 1].mean()
        return lat, lon
    except Exception:
        return np.nan, np.nan

# =========================
# APP STREAMLIT
# =========================
st.set_page_config(page_title="Oportunidades de negocio (CMadrid)", layout="wide")

from streamlit.components.v1 import html
html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

:root{
  --waze-blue: #00B8FF;
  --waze-blue-600: #0096D1;
  --waze-blue-700: #007CAB;
  --waze-cyan: #67E8F9;
  --accent-green:#22C55E;
  --accent-orange:#FF8A00;
  --accent-red:#FF5252;
  --text:#111827;
  --muted:#5B677A;
  --bg:#F4FAFF;
  --bg-card:#FFFFFF;
  --border:#CFE8FF;
  --focus: rgba(0,184,255,.24);
  --radius: 12px;
  --grad-primary: linear-gradient(180deg, var(--waze-blue), var(--waze-blue-600));
  --shadow-card: 0 6px 24px rgba(0,152,222,.08);
}

html, body, [class*="css"] { 
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; 
  color: var(--text); 
}

[data-testid="stAppViewContainer"]{ background:var(--bg); }
[data-testid="stHeader"]{ background:transparent; }
[data-testid="stSidebar"]{ background:var(--bg-card); border-right:1px solid var(--border); }

.sidebar-title{ 
  font-weight:700; 
  font-size:1rem; 
  margin:.5rem 0 .25rem; 
  color:var(--muted); 
  letter-spacing:.02em; 
  text-transform:uppercase; 
}

h1,h2,h3,h4,h5, p, .stMarkdown, label { color:var(--text); }

.stButton > button{
  background:var(--waze-blue); 
  border:1px solid var(--waze-blue-600); 
  color:#fff; 
  border-radius:12px;
  padding:.55rem .9rem; 
  font-weight:700; 
  transition: filter .2s ease, transform .04s ease;
  box-shadow: 0 8px 18px rgba(0,152,222,.18);
}

.stButton > button:hover{ filter:brightness(1.06); }
.stButton > button:active{ transform: translateY(1px); }

[data-baseweb="select"] > div,[data-baseweb="input"],[data-baseweb="textarea"]{
  background:var(--bg-card)!important; 
  color:var(--text)!important; 
  border:1.5px solid var(--border)!important; 
  border-radius:10px!important; 
  box-shadow:none!important;
}

[data-baseweb="select"]:hover > div,[data-baseweb="input"]:hover,[data-baseweb="textarea"]:hover{ 
  border-color:var(--waze-blue)!important; 
}

[data-baseweb="select"]:focus-within > div,[data-baseweb="input"]:focus-within,[data-baseweb="textarea"]:focus-within{
  border-color:var(--waze-blue-600)!important; 
  box-shadow:0 0 0 3px var(--focus)!important;
}

[data-baseweb="select"] svg{ fill:var(--muted); }

.streamlit-expanderHeader{ 
  background:#EAF7FF; 
  border:1.5px solid var(--border); 
  border-radius:12px; 
}

div[data-testid="stAlert"]{ 
  border:1.5px solid var(--waze-blue-600); 
  border-radius:12px; 
  color:#0B2530; 
}

.stDataFrame div[data-testid="stTable"], div[data-testid="stDataFrame"] div[role="grid"]{ 
  border:1px solid var(--border); 
  border-radius:8px; 
}

.loader { 
  width:22px; 
  height:22px; 
  border-radius:50%; 
  border:3px solid #9CA3AF; 
  border-top-color:#374151; 
  animation: spin 1s linear infinite; 
  display:inline-block; 
  vertical-align:middle; 
  margin-right:.5rem;
}

@keyframes spin { to { transform: rotate(360deg);} }

.card{ 
  background:var(--bg-card); 
  border:1px solid var(--border); 
  border-radius:var(--radius); 
  box-shadow: var(--shadow-card); 
}

.card .card-top{ 
  height:4px; 
  background:var(--grad-primary); 
  border-radius:12px 12px 0 0; 
}

.badge{ 
  display:inline-block; 
  padding:.3rem .6rem; 
  border-radius:9999px; 
  background:#E6F7FF; 
  color:#05506B; 
  font-weight:700; 
  font-size:.75rem; 
}

.chip{ 
  display:inline-flex; 
  align-items:center; 
  gap:.4rem; 
  padding:.35rem .6rem; 
  border:1px solid var(--border); 
  border-radius:9999px; 
  background:#F0FAFF; 
  color:#05506B; 
  font-weight:700; 
  font-size:.75rem; 
}

*::selection{ background:rgba(0,184,255,.18); }

/* MOBILE OPTIMIZATIONS */
@media (max-width: 768px) {
  /* Hide logo on mobile */
  a[aria-label="Ayuntamiento de Madrid (abre en nueva pestaña)"]{ 
    display:none !important; 
  }
  
  /* Full-width sidebar on mobile when open */
  [data-testid="stSidebar"] {
    width: 100vw !important;
    max-width: 100vw !important;
    padding: 1rem !important;
    z-index: 999999 !important;
  }
  
  /* Sidebar content adjustments */
  [data-testid="stSidebar"] > div {
    width: 100% !important;
    max-width: 100% !important;
  }
  
  /* Hide main content when sidebar is open on mobile */
  [data-testid="stSidebar"][aria-expanded="true"] ~ [data-testid="stAppViewContainer"] {
    display: none !important;
  }
  
  /* Mobile-friendly title */
  h1 { 
    font-size: 1.5rem !important; 
    line-height: 1.3 !important; 
    margin-bottom: 1rem !important;
  }
  
  /* Compact info card on mobile */
  .card {
    margin: 0.25rem 0 0.75rem !important;
  }
  
  .card div[style*="padding:12px"] {
    padding: 8px 10px !important;
    gap: 8px !important;
  }
  
  /* Smaller info icon */
  .card div[style*="min-width:34px"] {
    min-width: 28px !important;
    height: 28px !important;
    font-size: 0.8rem !important;
  }
  
  .sidebar-title {
    font-size: 1rem !important;
    margin: 1rem 0 0.5rem !important;
    text-align: center !important;
  }
  
  /* Mobile-friendly buttons in sidebar */
  [data-testid="stSidebar"] .stButton > button {
    padding: 0.75rem 1rem !important;
    font-size: 1rem !important;
    width: 100% !important;
    margin: 0.5rem 0 !important;
  }
  
  /* Mobile-friendly selectboxes in sidebar - FIXED FOR LONG ACTIVITY NAMES */
  [data-testid="stSidebar"] [data-baseweb="select"] > div {
    font-size: 0.9rem !important;
    padding: 0.75rem !important;
    min-height: 48px !important;
    white-space: normal !important;
    overflow: visible !important;
    height: auto !important;
    line-height: 1.3 !important;
  }
  
  /* Activity selectbox specific styling for long names */
  [data-testid="stSidebar"] [data-baseweb="select"] [role="option"] {
    white-space: normal !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    padding: 0.75rem 1rem !important;
    line-height: 1.4 !important;
    min-height: 44px !important;
    height: auto !important;
  }
  
  /* Selected value in dropdown should wrap */
  [data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="select-value"] {
    white-space: normal !important;
    word-wrap: break-word !important;
    overflow: visible !important;
    text-overflow: unset !important;
    line-height: 1.3 !important;
  }
  
  /* Dropdown container adjustments */
  [data-testid="stSidebar"] [data-baseweb="select"] [role="listbox"] {
    max-height: 300px !important;
    overflow-y: auto !important;
  }
  
  /* Mobile-friendly checkboxes in sidebar */
  [data-testid="stSidebar"] .stCheckbox {
    margin: 1rem 0 !important;
  }
  
  [data-testid="stSidebar"] .stCheckbox label {
    font-size: 1rem !important;
    padding: 0.5rem 0 !important;
  }
  
  /* Sidebar expander adjustments */
  [data-testid="stSidebar"] .streamlit-expanderHeader {
    padding: 1rem !important;
    font-size: 1rem !important;
    min-height: 48px !important;
  }
  
  /* Mobile-friendly buttons */
  .stButton > button {
    padding: 0.4rem 0.7rem !important;
    font-size: 0.9rem !important;
    width: 100% !important;
  }
  
  /* Mobile-friendly selectboxes */
  [data-baseweb="select"] > div {
    font-size: 0.9rem !important;
    padding: 0.4rem 0.7rem !important;
  }
  
  /* Compact expander */
  .streamlit-expanderHeader {
    padding: 0.5rem !important;
    font-size: 0.9rem !important;
  }
  
  /* Mobile-friendly map */
  iframe[title="streamlit_folium.st_folium"] {
    height: 400px !important;
  }
  
  /* Mobile table optimizations */
  .stDataFrame {
    font-size: 0.85rem !important;
  }
  
  .stDataFrame div[data-testid="stTable"] {
    max-height: 400px !important;
    overflow-y: auto !important;
  }
  
  /* Stack dataset info vertically on mobile */
  div:has(> p:contains("Dataset cargado")) p {
    font-size: 0.9rem !important;
    margin-bottom: 0.3rem !important;
  }
  
  /* Mobile-friendly subheaders */
  h2, h3 {
    font-size: 1.2rem !important;
    margin-bottom: 0.5rem !important;
  }
  
  /* Compact spacing */
  .block-container {
    padding: 1rem 0.5rem !important;
  }
}

/* Extra small screens (phones in portrait) */
@media (max-width: 480px) {
  h1 {
    font-size: 1.3rem !important;
  }
  
  .card div[style*="line-height:1.45"] {
    font-size: 0.85rem !important;
    line-height: 1.4 !important;
  }
  
  .card ul {
    margin: 0.2rem 0 0.2rem 0.8rem !important;
    font-size: 0.8rem !important;
  }
  
  /* Ultra-compact table for small screens */
  .stDataFrame div[data-testid="stTable"] {
    max-height: 300px !important;
  }
  
  .stDataFrame {
    font-size: 0.8rem !important;
  }
  
  /* Map adjustments for small screens */
  iframe[title="streamlit_folium.st_folium"] {
    height: 300px !important;
  }
}

/* Landscape orientation on mobile */
@media (max-width: 768px) and (orientation: landscape) {
  iframe[title="streamlit_folium.st_folium"] {
    height: 350px !important;
  }
  
  .stDataFrame div[data-testid="stTable"] {
    max-height: 250px !important;
  }
}

/* Touch-friendly improvements */
@media (pointer: coarse) {
  .stButton > button {
    min-height: 44px !important;
    padding: 0.6rem 1rem !important;
  }
  
  [data-baseweb="select"] > div {
    min-height: 44px !important;
    padding: 0.6rem !important;
  }
  
  .streamlit-expanderHeader {
    min-height: 44px !important;
    padding: 0.6rem !important;
  }
}
</style>
""", height=0)

# ---------- Título ----------
st.title("🗺️ Identificador de oportunidades de negocio de proximidad\n(Ayuntamiento de Madrid)")
st.markdown("""
<div class="card" style="margin:.5rem 0 1rem; padding:0;">
  <div class="card-top"></div>
  <div style="display:flex; gap:12px; padding:12px 14px; align-items:flex-start;
              background:#F0FAFF; border-radius:0 0 var(--radius) var(--radius); border-top:0;">
    <div style="min-width:34px; height:34px; border-radius:10px; 
                background:var(--grad-primary); display:flex; align-items:center; justify-content:center; 
                color:#fff; font-weight:800;">i</div>
    <div style="line-height:1.45;">
      <div style="font-weight:700; color:#0B2530; margin-bottom:.25rem;">¿Qué estás viendo?</div>
      <div style="color:var(--muted);">
        Este panel identifica <strong>oportunidades de negocio de proximidad</strong> en la Comunidad de Madrid.
        Selecciona una <em>actividad</em> a la izquierda para ver:
        <ul style="margin:.35rem 0 .25rem 1.1rem;">
          <li>Mapa con <strong>coropleta</strong> (barrio/distrito) según número de locales existentes.</li>
          <li>Tabla con zonas de <strong>baja competencia</strong> (potencial oportunidad).</li>
        </ul>
        <span style="font-size:.9rem; color:#325A6A;">
          Fuente: Dataset municipal obtenido del portal de datos abiertos del Ayuntamiento de Madrid enriquecido con coordenadas reales.
        </span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Logo flotante ----------
def embed_logo_top_center(path: str, width: int = 120, top_px: int = 8,
                          link: str = "https://datos.madrid.es/portal/site/egob") -> str:
    p = Path(path)
    if not p.exists():
        return ""
    mime, _ = mimetypes.guess_type(p.as_posix())
    mime = mime or f"image/{p.suffix[1:]}"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return dedent(f"""
    <a href="{link}" target="_blank" rel="noopener" aria-label="Ayuntamiento de Madrid (abre en nueva pestaña)"
       style="position:fixed; top:{top_px}px; left:50%; transform:translateX(-50%);
              z-index:2147483647; text-decoration:none;">
      <img alt="Ayuntamiento de Madrid"
           src="data:{mime};base64,{b64}"
           style="width:{width}px; height:auto;
                  background:#fff; padding:6px 8px;
                  border:1px solid var(--border); border-radius:10px;
                  box-shadow:0 2px 8px rgba(0,0,0,.08);" />
    </a>
    """)

# Only show logo if file exists
logo_path = None
for logo_file in ["ayto_madrid_logo.png", os.path.join("assets", "ayto_madrid_logo.png")]:
    if os.path.exists(logo_file):
        logo_path = Path(logo_file)
        break

if logo_path:
    st.markdown(embed_logo_top_center(str(logo_path), width=120, top_px=8), unsafe_allow_html=True)

# =========================
# 1) Carga de datos
# =========================
@st.cache_data(show_spinner=True)
def load_data(csv_path, sep=";"):
    df = pd.read_csv(csv_path, sep=sep, low_memory=False)
    
    # Debug: Show available columns
    st.write("**Columns found in CSV:**", list(df.columns))
    
    # Look for coordinate columns with different possible names
    coord_mapping = {
        'latitude': 'lat',
        'Latitude': 'lat', 
        'LATITUDE': 'lat',
        'lat_centroide': 'lat',
        'latitud': 'lat',
        'longitude': 'lon',
        'Longitude': 'lon',
        'LONGITUDE': 'lon', 
        'lon_centroide': 'lon',
        'longitud': 'lon'
    }
    
    # Rename coordinate columns if found
    for old_name, new_name in coord_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
            st.info(f"Renamed column '{old_name}' to '{new_name}'")
    
    # If still no lat/lon columns, create empty ones
    if 'lat' not in df.columns:
        df['lat'] = np.nan
        st.warning("No latitude column found. Created empty 'lat' column.")
    
    if 'lon' not in df.columns:
        df['lon'] = np.nan
        st.warning("No longitude column found. Created empty 'lon' column.")
    
    # Convert coordinates to numeric, handling any potential issues
    if 'lat' in df.columns:
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    if 'lon' in df.columns:
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    
    # Show coordinate info
    valid_coords = df.dropna(subset=['lat', 'lon'])
    st.info(f"Found {len(valid_coords)} rows with valid coordinates out of {len(df)} total rows")
    
    # Normalize text columns - adapted to your column names
    text_cols = ["desc_epigrafe", "desc_division", "desc_seccion",
                "desc_barrio_local", "desc_distrito_local", 
                "clase_vial_edificio", "desc_vial_edificio", "rotulo"]
    
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(normalize_str)
    
    return df

# Ensure dataset is available (download from Google Drive if needed)
csv_path = ensure_dataset_available()

with st.spinner("Cargando datos…"):
    st.markdown('<span class="loader"></span>Preparando tabla y filtros', unsafe_allow_html=True)
    df = load_data(csv_path, SEP)

# =========================
# 2) Completar lat/lon con centroides si es necesario
# =========================
barrios_geo, barrios_index = load_geojson(BARRIOS_GEOJSON_PATH, BARRIO_PROP_KEY)
distritos_geo, distritos_index = load_geojson(DISTRITOS_GEOJSON_PATH, DISTRITO_PROP_KEY)

# Fill missing coordinates with centroids (barrio → distrito)
mask_centroid = df["lat"].isna() | df["lon"].isna()
if mask_centroid.any() and (barrios_index or distritos_index):
    st.info(f"Completando {mask_centroid.sum():,} coordenadas faltantes usando centroides...")
    
    # Try barrio centroids first
    if barrios_index:
        to_fill = df[mask_centroid].copy()
        lats, lons = [], []
        for _, r in to_fill.iterrows():
            barrio = normalize_str(r.get("desc_barrio_local"))
            lat, lon = (np.nan, np.nan)
            if barrio and barrio in barrios_index:
                lat, lon = feature_centroid(barrios_index[barrio])
            lats.append(lat); lons.append(lon)
        filled = pd.Series(lats).notna() & pd.Series(lons).notna()
        df.loc[mask_centroid, "lat"] = np.where(filled, lats, df.loc[mask_centroid, "lat"])
        df.loc[mask_centroid, "lon"] = np.where(filled, lons, df.loc[mask_centroid, "lon"])

    # Try distrito centroids for remaining missing
    mask_centroid = df["lat"].isna() | df["lon"].isna()
    if mask_centroid.any() and distritos_index:
        to_fill = df[mask_centroid].copy()
        lats, lons = [], []
        for _, r in to_fill.iterrows():
            dist = normalize_str(r.get("desc_distrito_local"))
            lat, lon = (np.nan, np.nan)
            if dist and dist in distritos_index:
                lat, lon = feature_centroid(distritos_index[dist])
            lats.append(lat); lons.append(lon)
        filled = pd.Series(lats).notna() & pd.Series(lons).notna()
        df.loc[mask_centroid, "lat"] = np.where(filled, lats, df.loc[mask_centroid, "lat"])
        df.loc[mask_centroid, "lon"] = np.where(filled, lons, df.loc[mask_centroid, "lon"])

# Data quality info
st.write("**Dataset cargado:**")
st.write(f"- Total registros: **{len(df):,}**")
st.write(f"- Con coordenadas: **{len(df.dropna(subset=['lat', 'lon'])):,}**")
st.write(f"- Actividades únicas: **{df['desc_epigrafe'].nunique():,}**")

# =========================
# 3) Sidebar: filtros y opciones
# =========================
st.sidebar.markdown('<div class="sidebar-title">Filtros</div>', unsafe_allow_html=True)

epigrafe_col = "desc_epigrafe"
if epigrafe_col not in df.columns:
    st.error("No encuentro la columna `desc_epigrafe` para filtrar por actividad.")
    st.stop()

# Clean activities list
actividades = df[epigrafe_col].dropna().unique()
actividades = sorted([x for x in actividades if str(x).strip() and x != 'NAN'])

if not actividades:
    st.error("No se encontraron actividades válidas en el dataset.")
    st.stop()

actividad_sel = st.sidebar.selectbox("🔎 Actividad / tipo de negocio", actividades, index=0)

st.sidebar.markdown('<div class="sidebar-title">Mapa</div>', unsafe_allow_html=True)
base_layer = st.sidebar.selectbox("Capa base", ["OpenStreetMap", "CartoDB positron"], index=1)
pintar = st.sidebar.selectbox("Coropleta por:", ["Barrio", "Distrito", "Sin coropleta"], index=0)
show_points = st.sidebar.checkbox("Mostrar puntos individuales", value=True)

with st.expander("👀 Vista previa de datos"):
    st.dataframe(df.head(20), use_container_width=True)

# =========================
# 4) Filtrado por actividad
# =========================
# Always filter by activity (removed the checkbox logic)
df_plot = df[df[epigrafe_col] == normalize_str(actividad_sel)].copy()
subtitulo = f"Actividad seleccionada: **{actividad_sel.title()}**"

st.markdown(f"### {subtitulo}")
st.write(f"Locales a representar: **{len(df_plot.dropna(subset=['lat','lon'])):,}**")

# Agregaciones por barrio/distrito
grupo_barrio = df_plot.groupby("desc_barrio_local").size().rename("n_locales").to_frame() \
    if "desc_barrio_local" in df_plot.columns else None
grupo_distrito = df_plot.groupby("desc_distrito_local").size().rename("n_locales").to_frame() \
    if "desc_distrito_local" in df_plot.columns else None

# =========================
# 5) Mapa
# =========================
st.subheader("🗺️ Mapa interactivo")

center_lat, center_lon = 40.4168, -3.7038
tiles_name = "cartodbpositron" if base_layer == "CartoDB positron" else "openstreetmap"
m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=tiles_name)

cmap = linear.Blues_09

# Coropletas
if pintar == "Barrio" and barrios_geo is not None and grupo_barrio is not None and not grupo_barrio.empty:
    comp = grupo_barrio.copy()
    comp["nivel"] = quantile_labels(comp["n_locales"])
    df_choro = comp.reset_index().rename(columns={"desc_barrio_local": BARRIO_PROP_KEY})
    max_val = max(1, int(df_choro["n_locales"].max()))
    cmap.scale(0, max_val)
    folium.Choropleth(
        geo_data=barrios_geo,
        data=df_choro,
        columns=[BARRIO_PROP_KEY, "n_locales"],
        key_on=f"feature.properties.{BARRIO_PROP_KEY}",
        fill_color='Blues',
        fill_opacity=0.75, line_opacity=0.25,
        legend_name="Nº de locales (competencia)",
        nan_fill_opacity=0.1
    ).add_to(m)

elif pintar == "Distrito" and distritos_geo is not None and grupo_distrito is not None and not grupo_distrito.empty:
    comp = grupo_distrito.copy()
    comp["nivel"] = quantile_labels(comp["n_locales"])
    df_choro = comp.reset_index().rename(columns={"desc_distrito_local": DISTRITO_PROP_KEY})
    max_val = max(1, int(df_choro["n_locales"].max()))
    cmap.scale(0, max_val)
    folium.Choropleth(
        geo_data=distritos_geo,
        data=df_choro,
        columns=[DISTRITO_PROP_KEY, "n_locales"],
        key_on=f"feature.properties.{DISTRITO_PROP_KEY}",
        fill_color='Blues',
        fill_opacity=0.75, line_opacity=0.25,
        legend_name="Nº de locales (competencia)",
        nan_fill_opacity=0.1
    ).add_to(m)

# Puntos
if show_points and ("lat" in df_plot.columns) and ("lon" in df_plot.columns):
    pts = df_plot.dropna(subset=["lat", "lon"]).copy()
    if len(pts) > 0:
        mc = MarkerCluster().add_to(m)
        popup_cols = [c for c in ["rotulo", "desc_epigrafe", "desc_barrio_local", "desc_distrito_local"] if c in df_plot.columns]
        
        def build_popup(row):
            bits = []
            for c in popup_cols:
                val = row.get(c, "")
                if pd.notna(val) and str(val).strip():
                    bits.append(f"<b>{c.replace('_', ' ').title()}:</b> {str(val).title()}")
            return "<br>".join(bits) if bits else "Sin datos"
        
        for _, r in pts.iterrows():
            try:
                folium.Marker(
                    [float(r["lat"]), float(r["lon"])],
                    popup=folium.Popup(build_popup(r), max_width=300)
                ).add_to(mc)
            except Exception:
                pass

st_folium(m, width=None, height=650)

# =========================
# 6) TABLA: ZONAS CON OPORTUNIDAD
# =========================
st.subheader("🟢 Zonas con baja competencia (o sin presencia)")
opcion = st.selectbox("Nivel de análisis", ["Barrio", "Distrito"], index=0)

def compute_opportunities(level="Barrio"):
    level_col = "desc_barrio_local" if level == "Barrio" else "desc_distrito_local"
    
    # Get all unique areas from the actual data (not just those with current business type)
    if level == "Barrio":
        all_areas_in_data = df["desc_barrio_local"].dropna().unique()
    else:
        all_areas_in_data = df["desc_distrito_local"].dropna().unique()
    
    # Create a complete list with all areas from the dataset
    all_areas = pd.DataFrame({level_col: sorted(all_areas_in_data)})
    
    # Get areas that have the selected business type
    if level == "Barrio" and grupo_barrio is not None:
        existing_areas = grupo_barrio.reset_index().rename(columns={"index": level_col})
    elif level == "Distrito" and grupo_distrito is not None:
        existing_areas = grupo_distrito.reset_index().rename(columns={"index": level_col})
    else:
        existing_areas = pd.DataFrame(columns=[level_col, "n_locales"])
    
    # Merge ALL areas with business counts (areas with no businesses get 0)
    comp = all_areas.merge(existing_areas, how="left", on=level_col)
    comp["n_locales"] = comp["n_locales"].fillna(0).astype(int)
    
    # Calculate categories for the complete dataset
    if len(comp) > 0 and comp["n_locales"].max() > 0:
        comp["categoria"] = quantile_labels(comp["n_locales"])
    else:
        comp["categoria"] = "Sin datos"
    
    # Sort by number of competitors (ascending) so 0-competitor areas appear first
    comp = comp.sort_values(["n_locales", level_col], ascending=[True, True])
    
    return comp[[level_col, "n_locales", "categoria"]]

tabla_op = compute_opportunities(opcion)
st.dataframe(tabla_op, use_container_width=True, height=600)

st.markdown("""
**Interpretación**
- **n_locales** = cuántos negocios de esta actividad hay en la zona.  
- **Muy baja/Baja** = menor competencia → potencial oportunidad.  
- **Alta** = mayor competencia → quizá menos atractivo para negocio de proximidad.
""")