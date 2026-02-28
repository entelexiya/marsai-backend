import random
import uuid
from typing import List, Dict, Any
from mission_configs import MISSION_FILE_TEMPLATES

_sol_counter = {"mars": 847, "satellite": 1, "lunar": 1, "deepspace": 1}
_file_counter = [0]

# Realistic sensor ranges per mission
# These match exactly what each IsolationForest was trained on
MISSION_SENSOR_RANGES = {
    "mars": {
        # Real Mars MEDA ranges (Perseverance, NASA PDS)
        # Temp: extreme diurnal cycle -90°C (night) to -17°C (afternoon)
        # Pressure: 700-751 Pa seasonal cycle
        "temp_range": (-90, -17), "temp_std": 15,
        "pres_range": (700, 751), "pres_std": 5,
        "ci_base": 0.12, "ci_std": 0.06,
        "rad_range": (0.1, 0.5),
        "hum_range": (0.0, 0.02),
        "anomaly_temp_delta": 30,    # how much temp spikes on anomaly
        "anomaly_pres_delta": 40,
    },
    "satellite": {
        # Earth surface sensors (LEO satellite instruments)
        # Temp: surface temperature typical range
        # Pressure: sea-level atmospheric pressure
        "temp_range": (15, 35), "temp_std": 8,
        "pres_range": (980, 1040), "pres_std": 8,
        "ci_base": 0.10, "ci_std": 0.05,
        "rad_range": (0.0, 0.15),
        "hum_range": (0.3, 0.8),     # Earth has humidity!
        "anomaly_temp_delta": 25,    # wildfire = big temp spike
        "anomaly_pres_delta": 20,
    },
    "lunar": {
        # Moon surface: extreme vacuum, huge temp swings
        # Day: +120°C, Night: -170°C, no atmosphere (pressure ~0)
        "temp_range": (-170, 120), "temp_std": 60,
        "pres_range": (0, 0.00001), "pres_std": 0,
        "ci_base": 0.08, "ci_std": 0.04,
        "rad_range": (0.2, 0.9),     # high radiation on Moon
        "hum_range": (0.0, 0.0),     # no humidity on Moon
        "anomaly_temp_delta": 15,
        "anomaly_pres_delta": 0,
    },
    "deepspace": {
        # Deep space: near absolute zero, vacuum
        # Voyager-class: ~-270°C, essentially 0 pressure
        "temp_range": (-273, -260), "temp_std": 1,
        "pres_range": (0, 0), "pres_std": 0,
        "ci_base": 0.05, "ci_std": 0.03,
        "rad_range": (0.01, 0.3),
        "hum_range": (0.0, 0.0),
        "anomaly_temp_delta": 2,
        "anomaly_pres_delta": 0,
    },
}


def generate_file(mission: str = "mars") -> Dict[str, Any]:
    templates = MISSION_FILE_TEMPLATES.get(mission, MISSION_FILE_TEMPLATES["mars"])
    template = random.choice(templates)
    desc = random.choice(template["descriptions"])
    size = round(random.uniform(*template["size_range"]), 1)
    sol = _sol_counter[mission]
    _file_counter[0] += 1

    ranges = MISSION_SENSOR_RANGES.get(mission, MISSION_SENSOR_RANGES["mars"])

    # Generate normal sensor readings for this mission
    temp = round(random.uniform(*ranges["temp_range"]) + random.gauss(0, ranges["temp_std"] * 0.3), 2)
    pres = round(random.uniform(*ranges["pres_range"]) + random.gauss(0, ranges.get("pres_std", 1)), 4)
    ci   = round(min(1.0, max(0.0, ranges["ci_base"] + random.gauss(0, ranges["ci_std"]))), 3)
    rad  = round(random.uniform(*ranges["rad_range"]), 3)
    hum  = round(random.uniform(*ranges["hum_range"]), 4)

    sensor_data = {
        "temperature": temp,
        "pressure": pres,
        "chemical_index": ci,
        "radiation_level": rad,
        "humidity": hum,
    }

    # ~8% chance of real anomaly — stays within mission range but outlier
    if random.random() < 0.08:
        sensor_data["chemical_index"] = round(random.uniform(0.75, 1.0), 3)
        sensor_data["temperature"] = round(temp + random.gauss(0, ranges["anomaly_temp_delta"]), 2)
        if ranges.get("anomaly_pres_delta", 0) > 0:
            sensor_data["pressure"] = round(pres + random.gauss(0, ranges["anomaly_pres_delta"]), 2)
        # Radiation spike on anomaly
        sensor_data["radiation_level"] = round(min(1.0, rad * random.uniform(1.5, 3.0)), 3)

    ext = template.get("ext", "dat")
    prefix = template["name_prefix"]

    # Mission-specific file naming
    if mission == "satellite":
        name = f"{prefix}_PASS{sol}_{str(_file_counter[0]).zfill(4)}.{ext}"
    elif mission == "deepspace":
        name = f"{prefix}_{str(_file_counter[0]).zfill(6)}.{ext}"
    else:
        name = f"{prefix}_{sol}_{str(_file_counter[0]).zfill(4)}.{ext}"

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
        "mission": mission,
    }


def generate_batch(n: int = 8, mission: str = "mars") -> List[Dict[str, Any]]:
    return [generate_file(mission) for _ in range(n)]


def advance_sol(mission: str = "mars"):
    _sol_counter[mission] += 1