# -*- coding: utf-8 -*-
"""
Dashboard Multi-Bouees - Surveillance GPS Vieux Port de Marseille
Navigation par sidebar avec vue dashboard et details par bouee

Auteur: Dashboard IoT
Date: 2026-01-23
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

# Import des donnees fictives
from donnees_fictives import (
    generer_toutes_donnees_fictives,
    obtenir_positions_reference_fictives,
    calculer_distance_haversine,
    detecter_vol_potentiel,
    BOUEES_CONFIG
)

# =============================================================================
# CONFIGURATION DE LA PAGE STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="Surveillance GPS Bouees",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fichiers de stockage
FICHIER_REFERENCES = Path(__file__).parent / "positions_reference.json"
FICHIER_ALERTES = Path(__file__).parent / "historique_alertes.json"
RAYON_SECURITE = 500

# =============================================================================
# CSS GLOBAL
# =============================================================================
st.markdown("""
<style>
    /* Reduire les marges */
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* Header personnalise */
    .custom-header {
        background: linear-gradient(135deg, #1a237e 0%, #3949ab 100%);
        padding: 15px 25px;
        margin: -1rem -1rem 1rem -1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: white;
        border-radius: 0 0 10px 10px;
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

    /* Stats cards */
    .stats-row {
        display: flex;
        gap: 15px;
        margin-bottom: 15px;
        flex-wrap: wrap;
    }
    .stat-card {
        background: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px 20px;
        flex: 1;
        min-width: 120px;
        text-align: center;
    }
    .stat-value {
        font-size: 28px;
        font-weight: 700;
        color: #1a237e;
    }
    .stat-value.success { color: #27ae60; }
    .stat-value.warning { color: #f39c12; }
    .stat-value.alert { color: #e74c3c; }
    .stat-label {
        font-size: 12px;
        color: #555;
        text-transform: uppercase;
        margin-top: 5px;
    }

    /* Badges etat */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-disponible {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .badge-indisponible {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .badge-alerte {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
        animation: pulse 2s infinite;
    }
    .badge-inconnu {
        background: #e2e3e5;
        color: #383d41;
        border: 1px solid #d6d8db;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    /* Table des bouees */
    .bouee-table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }
    .bouee-table th {
        background: #1a237e;
        color: white;
        padding: 12px 15px;
        text-align: left;
        font-weight: 600;
    }
    .bouee-table td {
        padding: 12px 15px;
        border-bottom: 1px solid #eee;
    }
    .bouee-table tr:hover {
        background: #f5f5f5;
    }

    /* Legende */
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
    .legend-line {
        width: 20px;
        height: 3px;
        flex-shrink: 0;
    }

    /* Alerte vol */
    .alerte-vol {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .alerte-vol-icon {
        font-size: 32px;
    }
    .alerte-vol-content h3 {
        margin: 0 0 5px 0;
        font-size: 16px;
    }
    .alerte-vol-content p {
        margin: 0;
        font-size: 13px;
        opacity: 0.9;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a237e 0%, #283593 100%);
    }
    [data-testid="stSidebar"] .stButton button {
        width: 100%;
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        color: white;
        margin-bottom: 5px;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: rgba(255,255,255,0.2);
        border-color: rgba(255,255,255,0.3);
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
# DECODAGE PAYLOAD
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
# RECUPERATION DES DONNEES API
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


# =============================================================================
# PREPARATION DES DONNEES GLOBALES
# =============================================================================
def preparer_donnees_toutes_bouees(api_key, stream_id):
    """
    Prepare les donnees pour toutes les bouees (fictives + reelle).
    """
    # Donnees fictives
    donnees_fictives = generer_toutes_donnees_fictives()

    # Donnees API reelle
    donnees_api = recuperer_donnees_api(api_key, stream_id, 200)
    df_api = aplatir_donnees(donnees_api, stream_id) if donnees_api else pd.DataFrame()

    # Construire la liste complete des bouees
    toutes_bouees = {}

    # Ajouter les bouees fictives
    for bouee_id, data in donnees_fictives.items():
        dernier = data["dernier_point"]
        toutes_bouees[bouee_id] = {
            "id": bouee_id,
            "nom": data["nom"],
            "type": "FICTIVE",
            "etat": data["etat"],
            "position_reference": data["position_reference"],
            "latitude": dernier["latitude"] if dernier else None,
            "longitude": dernier["longitude"] if dernier else None,
            "temperature": dernier["temperature"] if dernier else None,
            "batterie": dernier["batterie"] if dernier else None,
            "tilt_alert": dernier["tilt_alert"] if dernier else False,
            "timestamp": dernier["timestamp"] if dernier else None,
            "historique_positions": data["historique_positions"],
            "vol_detecte": data["vol_detecte"],
            "distance_max": data["distance_max_reference"]
        }

    # Ajouter la bouee reelle
    capteur_reel_id = "urn:lo:nsid:lora:24E124745E033174"
    if not df_api.empty:
        df_gps = df_api.dropna(subset=["Latitude", "Longitude"])
        if not df_gps.empty:
            dernier = df_gps.iloc[-1]
            positions_ref = charger_positions_reference()
            ref = positions_ref.get(capteur_reel_id, {})

            # Construire historique pour capteur reel
            historique = []
            for _, row in df_gps.iterrows():
                historique.append({
                    "timestamp": row["Timestamp"].to_pydatetime() if pd.notna(row["Timestamp"]) else None,
                    "latitude": row["Latitude"],
                    "longitude": row["Longitude"],
                    "temperature": row["Temperature"] if pd.notna(row.get("Temperature")) else None,
                    "batterie": row["Batterie"] if pd.notna(row.get("Batterie")) else None,
                    "tilt_alert": row.get("Tilt_Alert", False)
                })

            # Determiner etat
            etat = "DISPONIBLE"
            if dernier.get("Tilt_Alert"):
                etat = "INDISPONIBLE"

            toutes_bouees[capteur_reel_id] = {
                "id": capteur_reel_id,
                "nom": ref.get("nom", "Capteur Reel"),
                "type": "REELLE",
                "etat": etat,
                "position_reference": {"latitude": ref.get("latitude", 43.302205), "longitude": ref.get("longitude", 5.378178)},
                "latitude": dernier["Latitude"],
                "longitude": dernier["Longitude"],
                "temperature": dernier["Temperature"] if pd.notna(dernier.get("Temperature")) else None,
                "batterie": dernier["Batterie"] if pd.notna(dernier.get("Batterie")) else None,
                "tilt_alert": dernier.get("Tilt_Alert", False),
                "timestamp": dernier["Timestamp"].to_pydatetime() if pd.notna(dernier["Timestamp"]) else None,
                "historique_positions": historique,
                "vol_detecte": False,
                "distance_max": 0
            }
    else:
        # Capteur reel sans donnees
        positions_ref = charger_positions_reference()
        ref = positions_ref.get(capteur_reel_id, {})
        toutes_bouees[capteur_reel_id] = {
            "id": capteur_reel_id,
            "nom": ref.get("nom", "Capteur Reel"),
            "type": "REELLE",
            "etat": "INCONNU",
            "position_reference": {"latitude": ref.get("latitude", 43.302205), "longitude": ref.get("longitude", 5.378178)},
            "latitude": None,
            "longitude": None,
            "temperature": None,
            "batterie": None,
            "tilt_alert": False,
            "timestamp": None,
            "historique_positions": [],
            "vol_detecte": False,
            "distance_max": 0
        }

    return toutes_bouees


# =============================================================================
# CREATION DES CARTES
# =============================================================================
def creer_carte_dashboard(toutes_bouees, positions_ref):
    """
    Cree la carte globale du dashboard avec toutes les bouees.
    """
    # Centre sur le Vieux Port de Marseille
    center = [43.2950, 5.3700]

    carte = folium.Map(location=center, zoom_start=15, tiles="CartoDB positron")

    # Ajouter chaque bouee
    for bouee_id, bouee in toutes_bouees.items():
        lat = bouee.get("latitude")
        lon = bouee.get("longitude")
        ref = bouee.get("position_reference", {})

        # Zone de securite
        if ref.get("latitude") and ref.get("longitude"):
            folium.Circle(
                location=[ref["latitude"], ref["longitude"]],
                radius=RAYON_SECURITE,
                color="#3498db", fill=True, fill_color="#3498db", fill_opacity=0.05,
                weight=1, dash_array="5, 5"
            ).add_to(carte)

        # Marqueur selon l'etat
        if lat and lon:
            etat = bouee.get("etat", "INCONNU")

            if etat == "DISPONIBLE":
                color = "green"
                icon_color = "#27ae60"
            elif etat == "INDISPONIBLE":
                color = "orange"
                icon_color = "#f39c12"
            elif etat == "ALERTE_VOL":
                color = "red"
                icon_color = "#e74c3c"
            else:
                color = "gray"
                icon_color = "#95a5a6"

            # Popup avec infos
            popup_html = f"""
            <div style="min-width:150px">
                <b>{bouee['nom']}</b><br>
                <small>{bouee_id[-12:]}</small><br>
                <hr style="margin:5px 0">
                Etat: <b>{etat}</b><br>
                {'Temp: ' + str(bouee['temperature']) + ' C<br>' if bouee.get('temperature') else ''}
                {'Batterie: ' + str(bouee['batterie']) + '%<br>' if bouee.get('batterie') else ''}
                {'<span style="color:red">VOL DETECTE</span>' if bouee.get('vol_detecte') else ''}
            </div>
            """

            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color=color, icon="ship", prefix="fa"),
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=f"{bouee['nom']} - {etat}"
            ).add_to(carte)

    # Ajuster la vue
    all_pts = []
    for bouee in toutes_bouees.values():
        if bouee.get("latitude") and bouee.get("longitude"):
            all_pts.append([bouee["latitude"], bouee["longitude"]])
        ref = bouee.get("position_reference", {})
        if ref.get("latitude") and ref.get("longitude"):
            all_pts.append([ref["latitude"], ref["longitude"]])

    if len(all_pts) > 1:
        lats = [p[0] for p in all_pts]
        lons = [p[1] for p in all_pts]
        carte.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=[50, 50])

    return carte


def creer_carte_detail_bouee(bouee):
    """
    Cree la carte detaillee pour une bouee avec historique des positions.
    """
    ref = bouee.get("position_reference", {})
    historique = bouee.get("historique_positions", [])

    # Centre sur la position de reference ou derniere position
    if ref.get("latitude") and ref.get("longitude"):
        center = [ref["latitude"], ref["longitude"]]
    elif bouee.get("latitude") and bouee.get("longitude"):
        center = [bouee["latitude"], bouee["longitude"]]
    else:
        center = [43.2950, 5.3700]

    carte = folium.Map(location=center, zoom_start=16, tiles="CartoDB positron")

    # Zone de securite
    if ref.get("latitude") and ref.get("longitude"):
        folium.Circle(
            location=[ref["latitude"], ref["longitude"]],
            radius=RAYON_SECURITE,
            color="#3498db", fill=True, fill_color="#3498db", fill_opacity=0.1,
            weight=2, dash_array="5, 5",
            tooltip="Zone de securite (500m)"
        ).add_to(carte)

        # Marqueur reference (ancre)
        folium.Marker(
            location=[ref["latitude"], ref["longitude"]],
            icon=folium.Icon(color="blue", icon="anchor", prefix="fa"),
            tooltip=f"Position de reference"
        ).add_to(carte)

    # Historique des positions (polyline avec gradient)
    if len(historique) > 1:
        # Determiner si vol detecte pour couleur
        vol_detecte = bouee.get("vol_detecte", False)

        # Creer des segments avec couleurs differentes
        positions = [(pos["latitude"], pos["longitude"]) for pos in historique if pos.get("latitude") and pos.get("longitude")]

        if len(positions) > 1:
            # Gradient de couleur: gris -> bleu -> rouge (si vol)
            nb_segments = len(positions) - 1

            for i in range(nb_segments):
                # Calcul de la couleur selon la position dans l'historique
                ratio = i / nb_segments

                if vol_detecte:
                    # Gradient gris -> rouge
                    if ratio < 0.5:
                        color = "#808080"  # Gris
                    elif ratio < 0.8:
                        color = "#f39c12"  # Orange
                    else:
                        color = "#e74c3c"  # Rouge
                else:
                    # Gradient gris -> bleu
                    if ratio < 0.5:
                        color = "#808080"  # Gris
                    else:
                        color = "#3498db"  # Bleu

                # Epaisseur proportionnelle a la fraicheur
                weight = 2 + int(ratio * 4)

                folium.PolyLine(
                    locations=[positions[i], positions[i + 1]],
                    color=color,
                    weight=weight,
                    opacity=0.5 + ratio * 0.5
                ).add_to(carte)

        # Marqueurs pour les points importants
        for i, pos in enumerate(historique):
            if not pos.get("latitude") or not pos.get("longitude"):
                continue

            # Premier point (ancien)
            if i == 0:
                folium.CircleMarker(
                    location=[pos["latitude"], pos["longitude"]],
                    radius=5,
                    color="#808080",
                    fill=True, fill_color="#808080", fill_opacity=0.7,
                    tooltip=f"Premier point - {pos['timestamp'].strftime('%d/%m %H:%M') if pos.get('timestamp') else 'N/A'}"
                ).add_to(carte)

            # Dernier point (recent)
            elif i == len(historique) - 1:
                etat = bouee.get("etat", "INCONNU")
                if etat == "DISPONIBLE":
                    color = "#27ae60"
                elif etat == "INDISPONIBLE":
                    color = "#f39c12"
                elif etat == "ALERTE_VOL":
                    color = "#e74c3c"
                else:
                    color = "#95a5a6"

                folium.CircleMarker(
                    location=[pos["latitude"], pos["longitude"]],
                    radius=10,
                    color=color,
                    fill=True, fill_color=color, fill_opacity=0.9,
                    weight=3,
                    tooltip=f"Position actuelle - {pos['timestamp'].strftime('%d/%m %H:%M') if pos.get('timestamp') else 'N/A'}"
                ).add_to(carte)

    # Ajuster la vue pour inclure tout l'historique
    all_pts = []
    if ref.get("latitude") and ref.get("longitude"):
        all_pts.append([ref["latitude"], ref["longitude"]])
    for pos in historique:
        if pos.get("latitude") and pos.get("longitude"):
            all_pts.append([pos["latitude"], pos["longitude"]])

    if len(all_pts) > 1:
        lats = [p[0] for p in all_pts]
        lons = [p[1] for p in all_pts]
        carte.fit_bounds([[min(lats) - 0.001, min(lons) - 0.001], [max(lats) + 0.001, max(lons) + 0.001]])

    return carte


# =============================================================================
# AFFICHAGE DASHBOARD PRINCIPAL
# =============================================================================
def afficher_dashboard_principal(toutes_bouees, positions_ref):
    """
    Affiche la vue dashboard avec toutes les bouees.
    """
    # Header
    st.markdown("""
    <div class="custom-header">
        <h1 class="header-title">Surveillance GPS Bouees - Dashboard</h1>
        <div class="header-status">
            <span class="status-badge">Connecte</span>
            <span>MAJ auto: 1 min</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats globales
    nb_total = len(toutes_bouees)
    nb_dispo = sum(1 for b in toutes_bouees.values() if b.get("etat") == "DISPONIBLE")
    nb_indispo = sum(1 for b in toutes_bouees.values() if b.get("etat") == "INDISPONIBLE")
    nb_vol = sum(1 for b in toutes_bouees.values() if b.get("etat") == "ALERTE_VOL")
    nb_inconnu = sum(1 for b in toutes_bouees.values() if b.get("etat") == "INCONNU")

    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-value">{nb_total}</div>
            <div class="stat-label">Total Bouees</div>
        </div>
        <div class="stat-card">
            <div class="stat-value success">{nb_dispo}</div>
            <div class="stat-label">Disponibles</div>
        </div>
        <div class="stat-card">
            <div class="stat-value warning">{nb_indispo}</div>
            <div class="stat-label">Indisponibles</div>
        </div>
        <div class="stat-card">
            <div class="stat-value alert">{nb_vol}</div>
            <div class="stat-label">Alertes Vol</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tableau des bouees
    st.markdown("### Liste des Bouees")

    # Construire les donnees du tableau
    data_table = []
    for bouee_id, bouee in toutes_bouees.items():
        etat = bouee.get("etat", "INCONNU")

        if etat == "DISPONIBLE":
            badge = '<span class="badge badge-disponible">DISPONIBLE</span>'
        elif etat == "INDISPONIBLE":
            badge = '<span class="badge badge-indisponible">INDISPONIBLE</span>'
        elif etat == "ALERTE_VOL":
            badge = '<span class="badge badge-alerte">ALERTE VOL</span>'
        else:
            badge = '<span class="badge badge-inconnu">INCONNU</span>'

        data_table.append({
            "Nom": bouee.get("nom", bouee_id[-12:]),
            "ID": bouee_id[-16:],
            "Etat": etat,
            "Type": bouee.get("type", "FICTIVE"),
            "Batterie": f"{bouee['batterie']}%" if bouee.get("batterie") else "-",
            "Temperature": f"{bouee['temperature']} C" if bouee.get("temperature") else "-",
            "Derniere MAJ": bouee["timestamp"].strftime("%d/%m %H:%M") if bouee.get("timestamp") else "-"
        })

    df_table = pd.DataFrame(data_table)

    # Afficher avec style
    st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Nom": st.column_config.TextColumn("Nom", width="medium"),
            "ID": st.column_config.TextColumn("ID", width="small"),
            "Etat": st.column_config.TextColumn("Etat", width="small"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Batterie": st.column_config.TextColumn("Batterie", width="small"),
            "Temperature": st.column_config.TextColumn("Temp.", width="small"),
            "Derniere MAJ": st.column_config.TextColumn("Derniere MAJ", width="medium")
        }
    )

    # Legende
    st.markdown("""
    <div class="legend-container">
        <div class="legend-title">Legende Carte</div>
        <div class="legend-items">
            <div class="legend-item"><div class="legend-dot" style="background:#27ae60"></div>Disponible</div>
            <div class="legend-item"><div class="legend-dot" style="background:#f39c12"></div>Indisponible (renversee)</div>
            <div class="legend-item"><div class="legend-dot" style="background:#e74c3c"></div>Alerte vol</div>
            <div class="legend-item"><div class="legend-dot" style="background:#95a5a6"></div>Etat inconnu</div>
            <div class="legend-item"><div class="legend-circle"></div>Zone securite (500m)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Carte globale
    st.markdown("### Carte Globale - Vieux Port de Marseille")
    carte = creer_carte_dashboard(toutes_bouees, positions_ref)
    st_folium(carte, use_container_width=True, height=500, returned_objects=[])


# =============================================================================
# AFFICHAGE DETAILS BOUEE
# =============================================================================
def afficher_details_bouee(bouee_id, bouee):
    """
    Affiche les details d'une bouee specifique.
    """
    nom = bouee.get("nom", bouee_id[-12:])
    etat = bouee.get("etat", "INCONNU")

    # Header
    st.markdown(f"""
    <div class="custom-header">
        <h1 class="header-title">{nom}</h1>
        <div class="header-status">
            <span>{bouee_id[-16:]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Alerte vol si detecte
    if bouee.get("vol_detecte"):
        st.markdown(f"""
        <div class="alerte-vol">
            <div class="alerte-vol-icon">&#9888;</div>
            <div class="alerte-vol-content">
                <h3>VOL POTENTIEL DETECTE</h3>
                <p>La bouee s'est deplacee de {bouee.get('distance_max', 0):.0f}m par rapport a sa position de reference.
                Un deplacement progressif a ete detecte dans l'historique.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Stats de la bouee
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if etat == "DISPONIBLE":
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value success">{etat}</div>
                <div class="stat-label">Etat</div>
            </div>
            """, unsafe_allow_html=True)
        elif etat == "INDISPONIBLE":
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value warning">{etat}</div>
                <div class="stat-label">Etat (Renversee)</div>
            </div>
            """, unsafe_allow_html=True)
        elif etat == "ALERTE_VOL":
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value alert">{etat}</div>
                <div class="stat-label">Etat</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{etat}</div>
                <div class="stat-label">Etat</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        temp = bouee.get("temperature")
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{temp if temp else '-'} C</div>
            <div class="stat-label">Temperature</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        bat = bouee.get("batterie")
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{bat if bat else '-'}%</div>
            <div class="stat-label">Batterie</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        ts = bouee.get("timestamp")
        ts_str = ts.strftime("%d/%m %H:%M") if ts else "-"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="font-size:18px">{ts_str}</div>
            <div class="stat-label">Derniere MAJ</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Legende historique
    st.markdown("""
    <div class="legend-container">
        <div class="legend-title">Legende Historique</div>
        <div class="legend-items">
            <div class="legend-item"><div class="legend-dot" style="background:#3498db"></div>Position reference</div>
            <div class="legend-item"><div class="legend-line" style="background:linear-gradient(90deg, #808080, #3498db)"></div>Historique normal</div>
            <div class="legend-item"><div class="legend-line" style="background:linear-gradient(90deg, #808080, #e74c3c)"></div>Historique vol</div>
            <div class="legend-item"><div class="legend-circle"></div>Zone securite (500m)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Carte avec historique
    st.markdown("### Carte et Historique des Positions")
    carte = creer_carte_detail_bouee(bouee)
    st_folium(carte, use_container_width=True, height=500, returned_objects=[])

    # Historique en tableau
    historique = bouee.get("historique_positions", [])
    if historique:
        st.markdown("### Historique des Positions")

        # Afficher les 20 derniers points
        data_hist = []
        for pos in historique[-20:][::-1]:  # Les plus recents en premier
            data_hist.append({
                "Date/Heure": pos["timestamp"].strftime("%d/%m/%Y %H:%M") if pos.get("timestamp") else "-",
                "Latitude": f"{pos['latitude']:.6f}" if pos.get("latitude") else "-",
                "Longitude": f"{pos['longitude']:.6f}" if pos.get("longitude") else "-",
                "Temp.": f"{pos['temperature']} C" if pos.get("temperature") else "-",
                "Batterie": f"{pos['batterie']}%" if pos.get("batterie") else "-",
                "Renverse": "Oui" if pos.get("tilt_alert") else "Non"
            })

        st.dataframe(pd.DataFrame(data_hist), use_container_width=True, hide_index=True, height=400)


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
def afficher_sidebar(toutes_bouees):
    """
    Affiche la sidebar avec la navigation.
    """
    with st.sidebar:
        st.markdown("## Navigation")

        # Bouton Dashboard
        if st.button("Dashboard", key="nav_dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
            st.session_state.bouee_selectionnee = None

        st.markdown("---")
        st.markdown("### Bouees")

        # Liste des bouees
        for bouee_id, bouee in toutes_bouees.items():
            nom = bouee.get("nom", bouee_id[-12:])
            etat = bouee.get("etat", "INCONNU")

            # Indicateur d'etat
            if etat == "DISPONIBLE":
                indicateur = "&#x1F7E2;"  # Cercle vert
            elif etat == "INDISPONIBLE":
                indicateur = "&#x1F7E0;"  # Cercle orange
            elif etat == "ALERTE_VOL":
                indicateur = "&#x1F534;"  # Cercle rouge
            else:
                indicateur = "&#x26AA;"  # Cercle gris

            if st.button(f"{nom}", key=f"nav_{bouee_id}", use_container_width=True):
                st.session_state.page = "detail"
                st.session_state.bouee_selectionnee = bouee_id

        st.markdown("---")
        st.markdown(f"<small>Total: {len(toutes_bouees)} bouees</small>", unsafe_allow_html=True)


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Auto-refresh toutes les 60 secondes
    st_autorefresh(interval=60000, limit=None, key="auto_refresh")

    # Initialiser session state
    if "page" not in st.session_state:
        st.session_state.page = "dashboard"
    if "bouee_selectionnee" not in st.session_state:
        st.session_state.bouee_selectionnee = None

    # Charger configuration API
    api_key, stream_id, erreurs = charger_configuration()
    if erreurs:
        st.error(f"Configuration manquante: {', '.join(erreurs)}")
        st.info("Creez un fichier .env avec API_KEY et STREAM_ID")
        # Continuer quand meme avec les donnees fictives
        api_key = api_key or ""
        stream_id = stream_id or ""

    # Charger positions de reference
    positions_ref = charger_positions_reference()

    # Preparer toutes les donnees
    toutes_bouees = preparer_donnees_toutes_bouees(api_key, stream_id)

    # Afficher sidebar
    afficher_sidebar(toutes_bouees)

    # Afficher la page selectionnee
    if st.session_state.page == "dashboard":
        afficher_dashboard_principal(toutes_bouees, positions_ref)

    elif st.session_state.page == "detail" and st.session_state.bouee_selectionnee:
        bouee_id = st.session_state.bouee_selectionnee
        if bouee_id in toutes_bouees:
            afficher_details_bouee(bouee_id, toutes_bouees[bouee_id])
        else:
            st.error(f"Bouee non trouvee: {bouee_id}")
            st.session_state.page = "dashboard"


if __name__ == "__main__":
    main()
