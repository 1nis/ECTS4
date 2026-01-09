# -*- coding: utf-8 -*-
"""
Dashboard de visualisation GPS - Milesight AT101
Connexion à l'API Orange Live Objects

Auteur: Dashboard IoT
Date: 2026-01-09
"""

import os
import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv

# =============================================================================
# CONFIGURATION DE LA PAGE STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="Dashboard GPS AT101",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
# =============================================================================
def charger_configuration():
    """
    Charge les variables d'environnement depuis le fichier .env
    Retourne un tuple (api_key, stream_id, erreur)
    """
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


# =============================================================================
# FONCTIONS DE RÉCUPÉRATION DES DONNÉES
# =============================================================================
@st.cache_data(ttl=60)  # Cache de 60 secondes pour éviter les appels excessifs
def recuperer_donnees_api(api_key: str, stream_id: str, limit: int) -> dict:
    """
    Récupère les données depuis l'API Orange Live Objects

    Args:
        api_key: Clé API pour l'authentification
        stream_id: Identifiant du flux de données
        limit: Nombre maximum de points à récupérer

    Returns:
        dict: Données JSON de l'API ou None en cas d'erreur
    """
    url = f"https://liveobjects.orange-business.com/api/v0/data/streams/{stream_id}"

    headers = {
        "X-API-KEY": api_key,
        "Accept": "application/json"
    }

    params = {
        "limit": limit
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API: {str(e)}")
        return None


def aplatir_donnees(donnees_brutes: list) -> pd.DataFrame:
    """
    Aplatit les données JSON imbriquées en DataFrame Pandas
    Version robuste qui ignore les erreurs de format
    """
    if not donnees_brutes:
        return pd.DataFrame()

    enregistrements = []
    erreurs_count = 0

    for item in donnees_brutes:
        try:
            # Sécurité 1 : Si l'item lui-même n'est pas un dictionnaire
            if not isinstance(item, dict):
                continue

            value = item.get("value", {})

            # Sécurité 2 : C'est ici que ça plantait.
            # Si 'value' est une chaîne (ex: donnée brute non décodée), on l'ignore.
            if not isinstance(value, dict):
                erreurs_count += 1
                continue

            # Le payload peut être directement dans value ou dans value['payload']
            payload = value.get("payload", value)
            if not isinstance(payload, dict):
                payload = {} # Sécurité supplémentaire

            # Extraction du timestamp
            timestamp_str = item.get("timestamp") or item.get("created")
            if timestamp_str:
                timestamp = pd.to_datetime(timestamp_str)
            else:
                timestamp = pd.NaT

            # Extraction des coordonnées GPS
            latitude = None
            longitude = None

            # Recherche des coordonnées dans les différents formats possibles
            if "latitude" in payload:
                latitude = payload.get("latitude")
                longitude = payload.get("longitude")
            elif "location" in payload:
                loc = payload.get("location", {})
                if isinstance(loc, dict):
                    latitude = loc.get("lat") or loc.get("latitude")
                    longitude = loc.get("lon") or loc.get("lng") or loc.get("longitude")
            elif "gps" in payload:
                gps = payload.get("gps", {})
                if isinstance(gps, dict):
                    latitude = gps.get("lat") or gps.get("latitude")
                    longitude = gps.get("lon") or gps.get("lng") or gps.get("longitude")

            # Extraction de la température
            temperature = (
                payload.get("temperature") or
                payload.get("temp") or
                payload.get("Temperature")
            )

            # Extraction du niveau de batterie
            batterie = (
                payload.get("battery") or
                payload.get("batterie") or
                payload.get("Battery") or
                payload.get("battery_level") or
                payload.get("bat")
            )

            enregistrements.append({
                "Timestamp": timestamp,
                "Temperature": temperature,
                "Latitude": latitude,
                "Longitude": longitude,
                "Batterie": batterie
            })

        except Exception:
            # On ignore silencieusement les erreurs individuelles pour ne pas bloquer l'affichage
            continue

    # Notification discrète si des données brutes ont été ignorées
    if erreurs_count > 0:
        st.toast(f"Note : {erreurs_count} messages non décodés ont été ignorés.", icon="ℹ️")

    # Création du DataFrame
    df = pd.DataFrame(enregistrements)

    # Conversion des types de données
    for col in ["Temperature", "Latitude", "Longitude", "Batterie"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Tri par timestamp décroissant
    if "Timestamp" in df.columns and not df.empty:
        df = df.sort_values("Timestamp", ascending=False).reset_index(drop=True)

    return df

# =============================================================================
# COMPOSANTS D'INTERFACE UTILISATEUR
# =============================================================================
def afficher_erreur_configuration(variables_manquantes: list):
    """
    Affiche un message d'erreur dans la sidebar si la configuration est incorrecte
    """
    st.sidebar.error("⚠️ Configuration requise")
    st.sidebar.markdown("""
    **Variables d'environnement manquantes:**
    """)
    for var in variables_manquantes:
        st.sidebar.markdown(f"- `{var}`")

    st.sidebar.markdown("""
    ---
    **Pour configurer l'application:**

    1. Créez un fichier `.env` à la racine du projet
    2. Ajoutez les lignes suivantes:
    ```
    API_KEY=votre_cle_api_ici
    STREAM_ID=votre_stream_id_ici
    ```
    3. Redémarrez l'application
    """)

    st.error("🔐 Configuration manquante - Veuillez configurer le fichier .env")
    st.stop()


def afficher_kpis(df: pd.DataFrame):
    """
    Affiche les KPIs en haut de page
    """
    st.markdown("### 📊 Indicateurs en temps réel")

    col1, col2, col3, col4 = st.columns(4)

    if df.empty:
        with col1:
            st.metric("Position", "N/A")
        with col2:
            st.metric("Température", "N/A")
        with col3:
            st.metric("Batterie", "N/A")
        with col4:
            st.metric("Dernière MAJ", "N/A")
        return

    # Dernière position
    derniere_ligne = df.iloc[0]

    with col1:
        lat = derniere_ligne.get("Latitude")
        lon = derniere_ligne.get("Longitude")
        if pd.notna(lat) and pd.notna(lon):
            st.metric(
                "📍 Dernière position",
                f"{lat:.5f}, {lon:.5f}"
            )
        else:
            st.metric("📍 Dernière position", "Non disponible")

    with col2:
        temp = derniere_ligne.get("Temperature")
        if pd.notna(temp):
            st.metric("🌡️ Température", f"{temp:.1f} °C")
        else:
            st.metric("🌡️ Température", "N/A")

    with col3:
        batterie = derniere_ligne.get("Batterie")
        if pd.notna(batterie):
            # Alerte couleur si batterie < 20%
            if batterie < 20:
                st.metric(
                    "🔋 Batterie",
                    f"{batterie:.0f} %",
                    delta="FAIBLE",
                    delta_color="inverse"
                )
                st.error("⚠️ Batterie faible!")
            elif batterie < 50:
                st.metric(
                    "🔋 Batterie",
                    f"{batterie:.0f} %",
                    delta="Moyenne",
                    delta_color="off"
                )
            else:
                st.metric(
                    "🔋 Batterie",
                    f"{batterie:.0f} %",
                    delta="OK",
                    delta_color="normal"
                )
        else:
            st.metric("🔋 Batterie", "N/A")

    with col4:
        timestamp = derniere_ligne.get("Timestamp")
        if pd.notna(timestamp):
            st.metric(
                "🕐 Dernière MAJ",
                timestamp.strftime("%d/%m/%Y %H:%M")
            )
        else:
            st.metric("🕐 Dernière MAJ", "N/A")


def afficher_carte(df: pd.DataFrame):
    """
    Affiche une carte interactive avec les positions GPS
    """
    st.markdown("### 🗺️ Carte des positions")

    # Filtrer les données avec coordonnées valides
    df_carte = df.dropna(subset=["Latitude", "Longitude"]).copy()

    if df_carte.empty:
        st.warning("Aucune donnée GPS valide à afficher sur la carte.")
        return

    # Renommer les colonnes pour st.map (requiert 'lat' et 'lon' ou 'latitude' et 'longitude')
    df_carte = df_carte.rename(columns={
        "Latitude": "lat",
        "Longitude": "lon"
    })

    # Affichage de la carte
    st.map(df_carte[["lat", "lon"]], zoom=12)

    # Informations supplémentaires
    st.caption(f"📍 {len(df_carte)} positions affichées sur la carte")


def afficher_graphique_temperature(df: pd.DataFrame):
    """
    Affiche un graphique temporel de la température
    """
    st.markdown("### 📈 Évolution de la température")

    # Filtrer les données avec température valide
    df_temp = df.dropna(subset=["Timestamp", "Temperature"]).copy()

    if df_temp.empty:
        st.warning("Aucune donnée de température disponible.")
        return

    # Trier par timestamp croissant pour le graphique
    df_temp = df_temp.sort_values("Timestamp", ascending=True)

    # Préparer les données pour le graphique
    df_chart = df_temp.set_index("Timestamp")[["Temperature"]]

    # Affichage du graphique
    st.line_chart(df_chart, y="Temperature", use_container_width=True)

    # Statistiques
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Min", f"{df_temp['Temperature'].min():.1f} °C")
    with col2:
        st.metric("Moyenne", f"{df_temp['Temperature'].mean():.1f} °C")
    with col3:
        st.metric("Max", f"{df_temp['Temperature'].max():.1f} °C")


def afficher_tableau_donnees(df: pd.DataFrame):
    """
    Affiche un tableau interactif des données brutes
    """
    st.markdown("### 📋 Données brutes")

    if df.empty:
        st.info("Aucune donnée à afficher.")
        return

    # Formatage du DataFrame pour l'affichage
    df_affichage = df.copy()

    # Formatage des colonnes
    if "Timestamp" in df_affichage.columns:
        df_affichage["Timestamp"] = df_affichage["Timestamp"].dt.strftime("%d/%m/%Y %H:%M:%S")

    if "Temperature" in df_affichage.columns:
        df_affichage["Temperature"] = df_affichage["Temperature"].apply(
            lambda x: f"{x:.1f} °C" if pd.notna(x) else "N/A"
        )

    if "Batterie" in df_affichage.columns:
        df_affichage["Batterie"] = df_affichage["Batterie"].apply(
            lambda x: f"{x:.0f} %" if pd.notna(x) else "N/A"
        )

    if "Latitude" in df_affichage.columns:
        df_affichage["Latitude"] = df_affichage["Latitude"].apply(
            lambda x: f"{x:.6f}" if pd.notna(x) else "N/A"
        )

    if "Longitude" in df_affichage.columns:
        df_affichage["Longitude"] = df_affichage["Longitude"].apply(
            lambda x: f"{x:.6f}" if pd.notna(x) else "N/A"
        )

    # Affichage du tableau interactif
    st.dataframe(
        df_affichage,
        use_container_width=True,
        hide_index=True
    )

    st.caption(f"Total: {len(df)} enregistrements")


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================
def main():
    """
    Fonction principale de l'application
    """
    # Titre de l'application
    st.title("🛰️ Dashboard GPS - Milesight AT101")
    st.markdown("*Visualisation des données du traceur GPS via Orange Live Objects*")
    st.markdown("---")

    # Chargement de la configuration
    api_key, stream_id, erreurs = charger_configuration()

    # Vérification de la configuration
    if erreurs:
        afficher_erreur_configuration(erreurs)
        return

    # Configuration de la sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")

    # Slider pour le nombre de points
    limit = st.sidebar.slider(
        "Nombre de points à récupérer",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Définit le nombre maximum d'enregistrements à charger depuis l'API"
    )

    # Bouton de rafraîchissement
    if st.sidebar.button("🔄 Rafraîchir les données", use_container_width=True):
        # Vider le cache pour forcer le rechargement
        st.cache_data.clear()
        st.rerun()

    # Informations de connexion
    st.sidebar.markdown("---")
    st.sidebar.success("✅ Connecté à Live Objects")
    st.sidebar.caption(f"Stream ID: {stream_id[:8]}...")

    # Récupération des données
    with st.spinner("Chargement des données depuis l'API..."):
        donnees_brutes = recuperer_donnees_api(api_key, stream_id, limit)

    if donnees_brutes is None:
        st.error("❌ Impossible de récupérer les données. Vérifiez votre connexion et vos identifiants.")
        return

    # Traitement des données
    df = aplatir_donnees(donnees_brutes)

    if df.empty:
        st.warning("⚠️ Aucune donnée disponible pour la période sélectionnée.")
        return

    # Affichage des KPIs
    afficher_kpis(df)

    st.markdown("---")

    # Disposition en deux colonnes pour la carte et le graphique
    col_carte, col_graph = st.columns(2)

    with col_carte:
        afficher_carte(df)

    with col_graph:
        afficher_graphique_temperature(df)

    st.markdown("---")

    # Tableau des données brutes
    afficher_tableau_donnees(df)

    # Footer
    st.markdown("---")
    st.caption(
        f"Dashboard GPS AT101 | "
        f"Données chargées: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | "
        f"Points affichés: {len(df)}"
    )


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================
if __name__ == "__main__":
    main()
