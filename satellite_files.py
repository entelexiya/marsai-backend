import random
import uuid
from typing import List, Dict, Any

FILE_TEMPLATES = [
    {
        "type": "IMG",
        "name_prefix": "ZCAM_SOL",
        "descriptions": [
            "Unusual reddish-brown mineral deposit detected near rock formation Delta-7, possible hematite concretions",
            "Routine terrain survey image, flat dust-covered basalt surface, no notable features",
            "Layered sedimentary rock outcrop with visible stratification — potential water-deposited origin",
            "Dust devil track visible in surface regolith, 340m in length heading northwest",
            "Close-up of rock abrasion target showing fine crystalline structure with sulfate veins",
            "Panoramic horizon scan, no anomalies detected, standard sol documentation",
        ],
        "size_range": (8, 45),
        "sensor_base": {"temperature": -23.0, "pressure": 729.0, "chemical_index": 0.15},
    },
    {
        "type": "CHEM",
        "name_prefix": "CHEMCAM_RDRS",
        "descriptions": [
            "LIBS spectroscopy result: elevated manganese oxide content — biosignature-relevant compound",
            "Standard basalt composition: SiO2 48%, FeO 18%, MgO 9% — no anomalies",
            "Perchlorate concentration spike detected: 0.6% by weight, significant for habitability research",
            "Calcium sulfate vein analysis: gypsum variant, consistent with ancient aqueous environment",
            "Routine rock classification complete: olivine-rich basalt, volcanic origin confirmed",
            "Organic carbon detection attempt — inconclusive signal, requires follow-up with SAM",
        ],
        "size_range": (2, 12),
        "sensor_base": {"temperature": -31.0, "pressure": 727.0, "chemical_index": 0.5},
    },
    {
        "type": "ATM",
        "name_prefix": "MEDA_ATMO",
        "descriptions": [
            "Sudden methane spike: 21 ppb above baseline at 18:32 local time — high scientific priority",
            "Standard atmospheric pressure reading: 730 Pa, temperature -45°C, wind 4.2 m/s NNE",
            "Dust storm onset detected — opacity index rising from 0.4 to 1.2 over 3 hours",
            "UV radiation flux measurement: solar irradiance 586 W/m², standard sol conditions",
            "Unusual pressure oscillation pattern detected — possible sub-surface gas release",
            "Routine morning atmospheric profile: inversion layer at 800m altitude",
        ],
        "size_range": (1, 8),
        "sensor_base": {"temperature": -45.0, "pressure": 712.0, "chemical_index": 0.3},
    },
    {
        "type": "SEISM",
        "name_prefix": "SEIS_EVENT",
        "descriptions": [
            "Marsquake magnitude 3.2 detected at 11:14 — strongest seismic event this sol, epicenter 847km",
            "Low-frequency seismic noise: likely thermal contraction of surface regolith at sunset",
            "High-frequency seismic signal detected — possible meteorite impact 200km north",
            "Background seismic monitoring: nominal levels, no events above M1.0 this period",
        ],
        "size_range": (3, 20),
        "sensor_base": {"temperature": -30.0, "pressure": 731.0, "chemical_index": 0.1},
    },
    {
        "type": "PIXL",
        "name_prefix": "PIXL_XRF",
        "descriptions": [
            "Potential biosignature indicator: organic sulfur compound cluster detected in rock matrix",
            "X-ray fluorescence: iron-magnesium silicate, standard mafic composition, low priority",
            "Phosphate mineral identification: apatite group — relevant to prebiotic chemistry research",
            "Carbonate mineral vein: siderite composition, formed in ancient CO2-rich water",
        ],
        "size_range": (5, 25),
        "sensor_base": {"temperature": -28.0, "pressure": 728.0, "chemical_index": 0.6},
    },
]

_sol_counter = [847]
_file_counter = [0]


def generate_file() -> Dict[str, Any]:
    template = random.choice(FILE_TEMPLATES)
    desc = random.choice(template["descriptions"])
    size = round(random.uniform(*template["size_range"]), 1)
    sol = _sol_counter[0]
    _file_counter[0] += 1

    # Add noise to sensor data
    base = template["sensor_base"]
    sensor_data = {
        "temperature": round(base["temperature"] + random.gauss(0, 5), 2),
        "pressure": round(base["pressure"] + random.gauss(0, 8), 2),
        "chemical_index": round(min(1.0, max(0.0, base["chemical_index"] + random.gauss(0, 0.15))), 3),
        "radiation_level": round(random.uniform(0.1, 0.9), 3),
        "humidity": round(random.uniform(0.0, 0.05), 4),
    }

    # Occasionally inject anomaly
    if random.random() < 0.08:
        sensor_data["chemical_index"] = round(random.uniform(0.82, 1.0), 3)
        sensor_data["temperature"] = round(base["temperature"] + random.gauss(0, 20), 2)

    name = f"{template['name_prefix']}_{sol}_{str(_file_counter[0]).zfill(4)}.{'jpg' if template['type'] == 'IMG' else 'dat' if template['type'] in ['ATM', 'SEISM'] else 'csv'}"

    return {
        "id": str(uuid.uuid4())[:8],
        "name": name,
        "type": template["type"],
        "size_mb": size,
        "description": desc,
        "sensor_data": sensor_data,
        "status": "pending",
        "priority": 1,
        "ai_score": 0.0,
        "decision_reason": "",
        "sol": sol,
    }


def generate_batch(n: int = 8) -> List[Dict[str, Any]]:
    return [generate_file() for _ in range(n)]


def advance_sol():
    _sol_counter[0] += 1
