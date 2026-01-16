# -*- coding: utf-8 -*-
"""
Carte interactive GPS - Milesight AT101
Surveillance de zone avec alertes

Auteur: Dashboard IoT
Date: 2026-01-16
"""

import os
import json
import math
import streamlit as st
import pandas as pd
import requests
import folium
from streamlit_folium import st_folium
from datetime import datetime
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh
from pathlib import Path

# =============================================================================
# CONFIGURATION DE LA PAGE STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="Surveillance GPS Bouées",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Fichiers de stockage
FICHIER_REFERENCES = Path(__file__).parent / "positions_reference.json"
FICHIER_ALERTES = Path(__file__).parent / "historique_alertes.json"
RAYON_SECURITE = 500

# =============================================================================
# CSS MINIMALISTE
# =============================================================================
st.markdown("""
<style>
    /* Réduire les marges */
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* Header personnalisé */
    .custom-header {
        background: linear-gradient(135deg, #1a237e 0%, #3949ab 100%);
        padding: 15px 25px;
        margin: -1rem -1rem 1rem -1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: white;
    }
    .header-title {
        font-size: 22px;
        font-weight: 600;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
        color: white;
    }
    .header-status {
        display: flex;
        align-items: center;
        gap: 15px;
        font-size: 13px;
        color: rgba(255,255,255,0.8);
    }
    .status-badge {
        background: rgba(76, 175, 80, 0.3);
        border: 1px solid #4caf50;
        padding: 4px 12px;
        border-radius: 15px;
        color: #a5d6a7;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #e8eaf6;
        padding: 5px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 25px;
        border-radius: 8px;
        font-weight: 500;
        color: #1a237e;
    }
    .stTabs [aria-selected="true"] {
        background: #1a237e !important;
        color: white !important;
    }

    /* Légende */
    .legend-container {
        background: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .legend-title {
        font-weight: 600;
        font-size: 14px;
        color: #333;
        margin-bottom: 12px;
    }
    .legend-items {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 13px;
        color: #333;
    }
    .legend-dot {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .legend-circle {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        border: 2px dashed #3498db;
        background: transparent;
        flex-shrink: 0;
    }

    /* Stats cards */
    .stats-row {
        display: flex;
        gap: 15px;
        margin-bottom: 15px;
    }
    .stat-card {
        background: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px 20px;
        flex: 1;
        text-align: center;
    }
    .stat-value {
        font-size: 28px;
        font-weight: 700;
        color: #1a237e;
    }
    .stat-value.alert {
        color: #e74c3c;
    }
    .stat-label {
        font-size: 12px;
        color: #555;
        text-transform: uppercase;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================
def charger_configuration():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    stream_id = os.getenv("STREAM_ID")
    erreurs = []
    if not api_key:
        erreurs.append("API_KEY")
    if not stream_id:
        erreurs.append("STREAM_ID")
    if erreurs:
        return None, None, erreurs
    return api_key, stream_id, None


def calculer_distance_haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def charger_positions_reference():
    if FICHIER_REFERENCES.exists():
        try:
            with open(FICHIER_REFERENCES, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}


def sauvegarder_positions_reference(positions):
    with open(FICHIER_REFERENCES, "w", encoding="utf-8") as f:
        json.dump(positions, f, indent=2, ensure_ascii=False)


def charger_historique_alertes():
    if FICHIER_ALERTES.exists():
        try:
            with open(FICHIER_ALERTES, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []


def sauvegarder_historique_alertes(alertes):
    with open(FICHIER_ALERTES, "w", encoding="utf-8") as f:
        json.dump(alertes, f, indent=2, ensure_ascii=False, default=str)


def ajouter_alerte(alertes, type_alerte, appareil, details):
    alerte = {
        "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "type": type_alerte,
        "appareil": appareil,
        "details": details
    }
    alertes.insert(0, alerte)
    if len(alertes) > 100:
        alertes = alertes[:100]
    sauvegarder_historique_alertes(alertes)
    return alertes


# =============================================================================
# DÉCODAGE PAYLOAD
# =============================================================================
def decoder_payload_milesight(payload_hex):
    result = {"battery": None, "temperature": None, "latitude": None, "longitude": None, "tilt_alert": None}
    if not payload_hex or not isinstance(payload_hex, str):
        return result
    try:
        payload_hex = payload_hex.replace(" ", "").replace("0x", "").lower()
        data = bytes.fromhex(payload_hex)
        i = 0
        while i < len(data):
            if i + 2 > len(data):
                break
            channel_id, channel_type = data[i], data[i + 1]
            i += 2
            if channel_type == 0x75 or (channel_id == 0x01 and channel_type == 0x01):
                if i + 1 <= len(data):
                    result["battery"] = data[i]
                    i += 1
            elif channel_type == 0x67:
                if i + 2 <= len(data):
                    result["temperature"] = int.from_bytes(data[i:i + 2], 'little', signed=True) / 10.0
                    i += 2
            elif channel_type == 0x88:
                if i + 8 <= len(data):
                    lat_raw = int.from_bytes(data[i:i + 4], 'little', signed=True)
                    lon_raw = int.from_bytes(data[i + 4:i + 8], 'little', signed=True)
                    if lat_raw != -1 and lon_raw != -1:
                        lat_val, lon_val = lat_raw / 1000000.0, lon_raw / 1000000.0
                        if -90 <= lat_val <= 90 and -180 <= lon_val <= 180:
                            result["latitude"], result["longitude"] = lat_val, lon_val
                    i += 8
            elif channel_id == 0x05 and channel_type == 0x00:
                if i + 1 <= len(data):
                    result["tilt_alert"] = (data[i] == 1)
                    i += 1
            else:
                break
    except:
        pass
    return result


# =============================================================================
# RÉCUPÉRATION DES DONNÉES
# =============================================================================
@st.cache_data(ttl=60)
def recuperer_donnees_api(api_key, stream_id, limit):
    url = f"https://liveobjects.orange-business.com/api/v0/data/streams/{stream_id}"
    try:
        response = requests.get(url, headers={"X-API-KEY": api_key}, params={"limit": limit}, timeout=30)
        response.raise_for_status()
        return response.json()
    except:
        return None


def aplatir_donnees(donnees_brutes, stream_id):
    if not donnees_brutes:
        return pd.DataFrame()
    enregistrements = []
    for item in donnees_brutes:
        try:
            if not isinstance(item, dict):
                continue
            value = item.get("value", {})
            appareil_id = item.get("streamId", stream_id)
            timestamp = pd.to_datetime(item.get("timestamp") or item.get("created"))
            lat, lon, temp, bat, tilt = None, None, None, None, None

            if isinstance(value, str):
                d = decoder_payload_milesight(value)
                lat, lon, temp, bat, tilt = d["latitude"], d["longitude"], d["temperature"], d["battery"], d["tilt_alert"]
            elif isinstance(value, dict):
                payload = value.get("payload", value)
                if isinstance(payload, str):
                    d = decoder_payload_milesight(payload)
                    lat, lon, temp, bat, tilt = d["latitude"], d["longitude"], d["temperature"], d["battery"], d["tilt_alert"]
                elif isinstance(payload, dict):
                    lat, lon = payload.get("latitude"), payload.get("longitude")
                    temp = payload.get("temperature") or payload.get("temp")
                    bat = payload.get("battery") or payload.get("batterie")
                    tilt = payload.get("tilt_alert")

            enregistrements.append({"Appareil": appareil_id, "Timestamp": timestamp, "Temperature": temp,
                                    "Latitude": lat, "Longitude": lon, "Batterie": bat, "Tilt_Alert": tilt})
        except:
            continue

    df = pd.DataFrame(enregistrements)
    for col in ["Temperature", "Latitude", "Longitude", "Batterie"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Timestamp" in df.columns and not df.empty:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        if df["Timestamp"].dt.tz is None:
            df["Timestamp"] = df["Timestamp"].dt.tz_localize("UTC")
        try:
            df["Timestamp"] = df["Timestamp"].dt.tz_convert("Europe/Paris")
        except:
            df["Timestamp"] = df["Timestamp"] + pd.Timedelta(hours=1)
        df = df.sort_values("Timestamp", ascending=True).reset_index(drop=True)
    return df


def definir_position_reference(df, positions_ref):
    df_gps = df.dropna(subset=["Latitude", "Longitude", "Appareil"])
    if df_gps.empty:
        return positions_ref
    modifie = False
    for appareil in df_gps["Appareil"].unique():
        if appareil not in positions_ref:
            df_app = df_gps[df_gps["Appareil"] == appareil].sort_values("Timestamp")
            if not df_app.empty:
                p = df_app.iloc[0]
                positions_ref[appareil] = {
                    "latitude": float(p["Latitude"]), "longitude": float(p["Longitude"]),
                    "date_reference": p["Timestamp"].strftime("%d/%m/%Y %H:%M:%S")
                }
                modifie = True
    if modifie:
        sauvegarder_positions_reference(positions_ref)
    return positions_ref


def verifier_sorties_zone(df, positions_ref, alertes):
    df = df.copy()
    df["Hors_Zone"], df["Distance_Reference"] = False, None

    alertes_existantes = {f"{a['appareil']}_{a['details'].get('latitude')}_{a['details'].get('longitude')}"
                         for a in alertes if a["type"] == "SORTIE_ZONE"}
    alertes_tilt = {f"TILT_{a['appareil']}_{a['details'].get('timestamp_point')}"
                   for a in alertes if a["type"] == "RENVERSEMENT"}

    for idx, row in df.dropna(subset=["Latitude", "Longitude", "Appareil"]).iterrows():
        appareil = row["Appareil"]
        if appareil in positions_ref:
            ref = positions_ref[appareil]
            distance = calculer_distance_haversine(ref["latitude"], ref["longitude"], row["Latitude"], row["Longitude"])
            df.at[idx, "Distance_Reference"] = distance
            if distance > RAYON_SECURITE:
                df.at[idx, "Hors_Zone"] = True
                key = f"{appareil}_{row['Latitude']}_{row['Longitude']}"
                if key not in alertes_existantes:
                    ts_str = row["Timestamp"].strftime("%d/%m/%Y %H:%M:%S") if pd.notna(row["Timestamp"]) else "N/A"
                    alertes = ajouter_alerte(alertes, "SORTIE_ZONE", appareil,
                                            {"latitude": row["Latitude"], "longitude": row["Longitude"],
                                             "distance": round(distance, 1), "timestamp_point": ts_str})

    for idx, row in df.iterrows():
        if row.get("Tilt_Alert") is True:
            ts_str = row["Timestamp"].strftime("%d/%m/%Y %H:%M:%S") if pd.notna(row.get("Timestamp")) else "N/A"
            key = f"TILT_{row.get('Appareil')}_{ts_str}"
            if key not in alertes_tilt:
                alertes = ajouter_alerte(alertes, "RENVERSEMENT", row.get("Appareil", "Inconnu"),
                                        {"latitude": row.get("Latitude"), "longitude": row.get("Longitude"),
                                         "timestamp_point": ts_str})
    return df, alertes


# =============================================================================
# CARTE FOLIUM
# =============================================================================
def creer_carte(df, positions_ref):
    df_carte = df.dropna(subset=["Latitude", "Longitude"]).copy()

    # Centre de la carte
    if positions_ref:
        lats = [p["latitude"] for p in positions_ref.values()]
        lons = [p["longitude"] for p in positions_ref.values()]
        center = [sum(lats) / len(lats), sum(lons) / len(lons)]
    elif not df_carte.empty:
        center = [df_carte["Latitude"].mean(), df_carte["Longitude"].mean()]
    else:
        center = [46.603354, 1.888334]

    carte = folium.Map(location=center, zoom_start=15, tiles="CartoDB positron")

    # Cercles de sûreté et ancres
    for appareil, ref in positions_ref.items():
        folium.Circle(
            location=[ref["latitude"], ref["longitude"]], radius=RAYON_SECURITE,
            color="#3498db", fill=True, fill_color="#3498db", fill_opacity=0.1,
            weight=2, dash_array="5, 5"
        ).add_to(carte)
        folium.Marker(
            location=[ref["latitude"], ref["longitude"]],
            icon=folium.Icon(color="blue", icon="anchor", prefix="fa"),
            tooltip=f"Référence: {appareil[-12:]}<br>{ref['date_reference']}"
        ).add_to(carte)

    # Points GPS
    for _, row in df_carte.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        tilt, hors_zone = row.get("Tilt_Alert"), row.get("Hors_Zone", False)

        if hors_zone and tilt:
            color = "#9b59b6"
        elif hors_zone:
            color = "#e74c3c"
        elif tilt:
            color = "#f39c12"
        else:
            color = "#27ae60"

        tooltip = []
        if pd.notna(row.get("Timestamp")):
            tooltip.append(f"<b>{row['Timestamp'].strftime('%d/%m %H:%M:%S')}</b>")
        tooltip.append(f"📍 {lat:.5f}, {lon:.5f}")
        if pd.notna(row.get("Temperature")):
            tooltip.append(f"🌡️ {row['Temperature']:.1f}°C")
        if pd.notna(row.get("Batterie")):
            tooltip.append(f"🔋 {row['Batterie']:.0f}%")
        if pd.notna(row.get("Distance_Reference")):
            tooltip.append(f"📏 {row['Distance_Reference']:.0f}m")

        folium.CircleMarker(
            location=[lat, lon], radius=6, color=color,
            fill=True, fill_color=color, fill_opacity=0.8, weight=2,
            tooltip="<br>".join(tooltip)
        ).add_to(carte)

    # Ajuster la vue
    all_pts = [[ref["latitude"], ref["longitude"]] for ref in positions_ref.values()]
    all_pts += [[r["Latitude"], r["Longitude"]] for _, r in df_carte.iterrows()]
    if len(all_pts) > 1:
        lats, lons = [p[0] for p in all_pts], [p[1] for p in all_pts]
        carte.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=[30, 30])

    return carte


# =============================================================================
# MAIN
# =============================================================================
def main():
    st_autorefresh(interval=60000, limit=None, key="auto_refresh")

    # Header
    st.markdown("""
    <div class="custom-header">
        <h1 class="header-title">🗺️ Surveillance GPS Bouées</h1>
        <div class="header-status">
            <span class="status-badge">● Connecté</span>
            <span>MAJ auto: 1 min</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Config
    api_key, stream_id, erreurs = charger_configuration()
    if erreurs:
        st.error(f"Configuration manquante: {', '.join(erreurs)}")
        st.stop()

    # Données
    positions_ref = charger_positions_reference()
    alertes = charger_historique_alertes()

    donnees = recuperer_donnees_api(api_key, stream_id, 200)
    if donnees is None:
        st.error("Erreur de connexion à l'API")
        st.stop()

    df = aplatir_donnees(donnees, stream_id)
    if df.empty:
        st.warning("Aucune donnée disponible")
        st.stop()

    positions_ref = definir_position_reference(df, positions_ref)
    df, alertes = verifier_sorties_zone(df, positions_ref, alertes)

    # Stats
    df_gps = df.dropna(subset=["Latitude", "Longitude"])
    nb_hors = (df["Hors_Zone"] == True).sum()
    nb_tilt = (df["Tilt_Alert"] == True).sum()

    # Tabs
    tab_carte, tab_alertes = st.tabs(["🗺️ Carte temps réel", f"🚨 Alertes ({len(alertes)})"])

    with tab_carte:
        # Stats
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-value">{len(df_gps)}</div>
                <div class="stat-label">Points GPS</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {'alert' if nb_hors > 0 else ''}">{nb_hors}</div>
                <div class="stat-label">Hors zone</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {'alert' if nb_tilt > 0 else ''}">{nb_tilt}</div>
                <div class="stat-label">Renversements</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{df_gps['Timestamp'].max().strftime('%H:%M') if not df_gps.empty and pd.notna(df_gps['Timestamp'].max()) else 'N/A'}</div>
                <div class="stat-label">Dernière MAJ</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Légende
        st.markdown("""
        <div class="legend-container">
            <div class="legend-title">Légende</div>
            <div class="legend-items">
                <div class="legend-item"><div class="legend-dot" style="background:#3498db"></div>Référence</div>
                <div class="legend-item"><div class="legend-dot" style="background:#27ae60"></div>OK</div>
                <div class="legend-item"><div class="legend-dot" style="background:#f39c12"></div>Renversement</div>
                <div class="legend-item"><div class="legend-dot" style="background:#e74c3c"></div>Hors zone</div>
                <div class="legend-item"><div class="legend-dot" style="background:#9b59b6"></div>Hors zone + Renversement</div>
                <div class="legend-item"><div class="legend-circle"></div>Zone sûreté (1km)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Carte
        if not df_gps.empty or positions_ref:
            carte = creer_carte(df, positions_ref)
            st_folium(carte, use_container_width=True, height=550, returned_objects=[])
        else:
            st.warning("Aucune donnée GPS valide")

    with tab_alertes:
        st.markdown("### Historique des alertes")

        if not alertes:
            st.info("Aucune alerte enregistrée")
        else:
            # Filtre
            filtre = st.selectbox("Filtrer", ["Toutes", "Sortie zone", "Renversement"], key="filtre_alertes")

            alertes_filtrees = alertes
            if filtre == "Sortie zone":
                alertes_filtrees = [a for a in alertes if a["type"] == "SORTIE_ZONE"]
            elif filtre == "Renversement":
                alertes_filtrees = [a for a in alertes if a["type"] == "RENVERSEMENT"]

            # Tableau
            if alertes_filtrees:
                data = []
                for a in alertes_filtrees:
                    data.append({
                        "Date": a["timestamp"],
                        "Type": "🔴 Sortie zone" if a["type"] == "SORTIE_ZONE" else "🟠 Renversement",
                        "Appareil": a["appareil"][-16:],
                        "Détails": f"Distance: {a['details'].get('distance', '-')}m" if a["type"] == "SORTIE_ZONE" else "Bouée renversée",
                        "Heure point": a["details"].get("timestamp_point", "-")
                    })
                st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True, height=450)
                st.caption(f"Total: {len(alertes_filtrees)} alertes")


if __name__ == "__main__":
    main()
