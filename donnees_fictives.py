# -*- coding: utf-8 -*-
"""
Generateur de donnees fictives pour bouees du Vieux Port de Marseille
"""

import random
import math
from datetime import datetime, timedelta

# =============================================================================
# CONFIGURATION DES BOUEES FICTIVES
# =============================================================================
BOUEES_CONFIG = {
    "BOUEE-VP-001": {
        "nom": "Quai des Belges",
        "position_reference": {"latitude": 43.2951, "longitude": 5.3743},
        "etat": "OK",  # OK, RENVERSEE, VOLEE
        "batterie_base": 85,
        "temperature_base": 15.5
    },
    "BOUEE-VP-002": {
        "nom": "Fort Saint-Jean",
        "position_reference": {"latitude": 43.2965, "longitude": 5.3622},
        "etat": "OK",
        "batterie_base": 92,
        "temperature_base": 14.8
    },
    "BOUEE-VP-003": {
        "nom": "Quai du Port",
        "position_reference": {"latitude": 43.2943, "longitude": 5.3687},
        "etat": "RENVERSEE",
        "batterie_base": 78,
        "temperature_base": 15.2
    },
    "BOUEE-VP-004": {
        "nom": "Quai de Rive Neuve",
        "position_reference": {"latitude": 43.2926, "longitude": 5.3696},
        "etat": "VOLEE",
        "batterie_base": 65,
        "temperature_base": 16.0
    }
}

# =============================================================================
# FONCTIONS DE GENERATION
# =============================================================================
def generer_variation_position(lat_base, lon_base, rayon_max_m=30):
    """
    Genere une petite variation autour de la position de base.
    Simule le mouvement naturel d'une bouee (vagues, courant).
    """
    # Conversion approximative: 1 degre lat ~= 111km, 1 degre lon ~= 111km * cos(lat)
    lat_offset = (random.uniform(-rayon_max_m, rayon_max_m) / 111000)
    lon_offset = (random.uniform(-rayon_max_m, rayon_max_m) / (111000 * math.cos(math.radians(lat_base))))
    return lat_base + lat_offset, lon_base + lon_offset


def generer_deplacement_vol(lat_base, lon_base, index, total_points):
    """
    Genere un deplacement progressif simulant un vol.
    La bouee s'eloigne progressivement de sa position de reference.
    """
    # Direction du deplacement (vers le sud-est)
    progression = index / total_points

    # Deplacement progressif jusqu'a 500m
    distance_m = 20 + (progression * 480)

    # Direction: vers le sud-est (augmente lat et lon legerement)
    angle_rad = math.radians(135 + random.uniform(-10, 10))  # ~Sud-Est

    lat_offset = (distance_m * math.cos(angle_rad)) / 111000
    lon_offset = (distance_m * math.sin(angle_rad)) / (111000 * math.cos(math.radians(lat_base)))

    # Ajouter du bruit
    lat_noise = random.uniform(-5, 5) / 111000
    lon_noise = random.uniform(-5, 5) / (111000 * math.cos(math.radians(lat_base)))

    return lat_base + lat_offset + lat_noise, lon_base + lon_offset + lon_noise


def generer_historique_positions(bouee_id, config, nb_points=50):
    """
    Genere l'historique des positions pour une bouee.
    """
    positions = []
    lat_base = config["position_reference"]["latitude"]
    lon_base = config["position_reference"]["longitude"]
    etat = config["etat"]

    # Timestamps sur les 7 derniers jours
    now = datetime.now()

    for i in range(nb_points):
        # Timestamp: reparti sur 7 jours, du plus ancien au plus recent
        heures_passees = int((nb_points - 1 - i) * (7 * 24 / nb_points))
        timestamp = now - timedelta(hours=heures_passees)

        # Position selon l'etat
        if etat == "VOLEE":
            lat, lon = generer_deplacement_vol(lat_base, lon_base, i, nb_points)
        else:
            # Mouvement naturel (vagues)
            lat, lon = generer_variation_position(lat_base, lon_base, rayon_max_m=25)

        # Temperature avec variation journaliere
        heure = timestamp.hour
        variation_temp = math.sin((heure - 6) * math.pi / 12) * 2  # +/- 2 degres
        temperature = config["temperature_base"] + variation_temp + random.uniform(-0.5, 0.5)

        # Batterie qui diminue legerement
        batterie = max(60, config["batterie_base"] - (i * 0.1) + random.uniform(-2, 2))

        # Tilt alert pour bouee renversee (seulement les derniers points)
        tilt_alert = False
        if etat == "RENVERSEE" and i >= nb_points - 10:
            tilt_alert = True

        positions.append({
            "timestamp": timestamp,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "temperature": round(temperature, 1),
            "batterie": round(batterie),
            "tilt_alert": tilt_alert
        })

    return positions


def calculer_distance_haversine(lat1, lon1, lat2, lon2):
    """Calcule la distance en metres entre deux points GPS."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def detecter_vol_potentiel(positions, position_reference):
    """
    Detecte si la bouee s'eloigne progressivement de sa position de reference.
    Retourne True si deplacement anormal detecte.
    """
    if len(positions) < 5:
        return False, 0, []

    lat_ref = position_reference["latitude"]
    lon_ref = position_reference["longitude"]

    # Calculer les distances pour les N derniers points
    distances = []
    for pos in positions[-20:]:  # 20 derniers points
        dist = calculer_distance_haversine(lat_ref, lon_ref, pos["latitude"], pos["longitude"])
        distances.append(dist)

    # Verifier tendance croissante
    if len(distances) >= 5:
        tendance = sum(1 for i in range(1, len(distances)) if distances[i] > distances[i-1])
        ratio_croissance = tendance / (len(distances) - 1)

        max_distance = max(distances)

        # Alerte si tendance croissante (>60%) et distance > 100m
        if ratio_croissance > 0.6 and max_distance > 100:
            return True, max_distance, distances

    return False, max(distances) if distances else 0, distances


def obtenir_etat_bouee(bouee_id, config, positions):
    """
    Determine l'etat actuel d'une bouee.
    Retourne: 'DISPONIBLE', 'INDISPONIBLE', 'ALERTE_VOL'
    """
    if not positions:
        return "INCONNU"

    dernier_point = positions[-1]

    # Verifier tilt alert (renversement)
    if dernier_point.get("tilt_alert"):
        return "INDISPONIBLE"

    # Verifier deplacement anormal (vol)
    vol_detecte, distance, _ = detecter_vol_potentiel(positions, config["position_reference"])
    if vol_detecte:
        return "ALERTE_VOL"

    return "DISPONIBLE"


def generer_toutes_donnees_fictives():
    """
    Genere les donnees completes pour toutes les bouees fictives.
    """
    donnees = {}

    for bouee_id, config in BOUEES_CONFIG.items():
        positions = generer_historique_positions(bouee_id, config)
        etat = obtenir_etat_bouee(bouee_id, config, positions)

        vol_detecte, distance_max, distances = detecter_vol_potentiel(
            positions, config["position_reference"]
        )

        donnees[bouee_id] = {
            "id": bouee_id,
            "nom": config["nom"],
            "position_reference": config["position_reference"],
            "etat": etat,
            "etat_simule": config["etat"],
            "historique_positions": positions,
            "dernier_point": positions[-1] if positions else None,
            "vol_detecte": vol_detecte,
            "distance_max_reference": round(distance_max, 1),
            "distances_historique": distances
        }

    return donnees


def obtenir_positions_reference_fictives():
    """
    Retourne les positions de reference pour toutes les bouees fictives.
    Format compatible avec positions_reference.json
    """
    positions = {}
    for bouee_id, config in BOUEES_CONFIG.items():
        positions[bouee_id] = {
            "latitude": config["position_reference"]["latitude"],
            "longitude": config["position_reference"]["longitude"],
            "date_reference": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "nom": config["nom"]
        }
    return positions


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("Generation des donnees fictives...")
    donnees = generer_toutes_donnees_fictives()

    for bouee_id, data in donnees.items():
        print(f"\n{data['nom']} ({bouee_id}):")
        print(f"  Etat: {data['etat']}")
        print(f"  Positions: {len(data['historique_positions'])}")
        print(f"  Dernier point: {data['dernier_point']}")
        print(f"  Vol detecte: {data['vol_detecte']}")
        print(f"  Distance max: {data['distance_max_reference']}m")
