"""Seed the database with 5 demo AOIs: two suspected mining zones, two suspected logging/
deforestation-front zones, and one control plot inside a strictly protected reserve. Run with:
python scripts/seed_zones.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlmodel import Session, select

from app.db import Zone, engine, init_db

DEMO_ZONES = [
    {
        "name": "Kambove Mining Watch",
        "zone_type": "mining",
        "centroid_lat": -10.8746,
        "centroid_lon": 26.5951,
        "aoi_geojson": json.dumps(
            {
                "type": "Polygon",
                "coordinates": [[[26.58, -10.88], [26.61, -10.88], [26.61, -10.86], [26.58, -10.86], [26.58, -10.88]]],
            }
        ),
    },
    {
        "name": "Tapajos Logging Corridor",
        "zone_type": "logging",
        "centroid_lat": -4.2661,
        "centroid_lon": -55.9836,
        "aoi_geojson": json.dumps(
            {
                "type": "Polygon",
                "coordinates": [[[-56.0, -4.28], [-55.96, -4.28], [-55.96, -4.25], [-56.0, -4.25], [-56.0, -4.28]]],
            }
        ),
    },
    {
        # Interior of Garamba National Park (DRC, UNESCO site, ~5,200 km^2), remote northern
        # sector: low patrol pressure, no agricultural encroachment on record. Replaces the old
        # "Kabaleqa Control Forest" AOI, which turned out not to be a stable control (see Stage
        # 3/4 backtest: 2024-07-15 showed a real -4.26 sigma NDVI loss there).
        "name": "Garamba Reference Plot",
        "zone_type": "control",
        "centroid_lat": 4.150,
        "centroid_lon": 29.550,
        "aoi_geojson": json.dumps(
            {
                "type": "Polygon",
                # ~1km x 1km box centered on the point above.
                "coordinates": [
                    [
                        [29.5455, 4.1455],
                        [29.5545, 4.1455],
                        [29.5545, 4.1545],
                        [29.5455, 4.1545],
                        [29.5455, 4.1455],
                    ]
                ],
            }
        ),
    },
    {
        # Southern Para deforestation frontier along the BR-163 highway corridor, near Novo
        # Progresso - a documented active deforestation front (INPE DETER publishes monthly
        # alerts here), unlike the existing Tapajos AOI, which sits inside the largely-protected
        # Tapajos National Forest interior.
        "name": "Novo Progresso Frontier",
        "zone_type": "logging",
        "centroid_lat": -7.150,
        "centroid_lon": -55.400,
        "aoi_geojson": json.dumps(
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-55.42, -7.17],
                        [-55.38, -7.17],
                        [-55.38, -7.13],
                        [-55.42, -7.13],
                        [-55.42, -7.17],
                    ]
                ],
            }
        ),
    },
    {
        # Illegal garimpo (artisanal gold mining) belt inside Yanomami Indigenous Territory,
        # Roraima - extensively documented by satellite monitoring, with a high VIIRS
        # night-light signal from dredge and camp generators.
        "name": "Yanomami Mining Belt",
        "zone_type": "mining",
        "centroid_lat": 3.500,
        "centroid_lon": -63.000,
        "aoi_geojson": json.dumps(
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-63.02, 3.48],
                        [-62.98, 3.48],
                        [-62.98, 3.52],
                        [-63.02, 3.52],
                        [-63.02, 3.48],
                    ]
                ],
            }
        ),
    },
]


def seed() -> None:
    """INPUTS: none. OUTPUTS: none; inserts the 5 demo zones if they don't already exist by name."""
    init_db()
    with Session(engine) as session:
        for zone_data in DEMO_ZONES:
            existing = session.exec(select(Zone).where(Zone.name == zone_data["name"])).first()
            if existing:
                print(f"Zone '{zone_data['name']}' already exists, skipping.")
                continue
            session.add(Zone(**zone_data))
            print(f"Inserting zone '{zone_data['name']}' ({zone_data['zone_type']})")
        session.commit()


if __name__ == "__main__":
    seed()
