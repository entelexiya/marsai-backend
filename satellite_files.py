import random
import uuid
from typing import List, Dict, Any
from mission_configs import MISSION_FILE_TEMPLATES

_sol_counter = {"mars": 847, "satellite": 1, "lunar": 1, "deepspace": 1}
_file_counter = [0]


def generate_file(mission: str = "mars") -> Dict[str, Any]:
    templates = MISSION_FILE_TEMPLATES.get(mission, MISSION_FILE_TEMPLATES["mars"])
    template = random.choice(templates)
    desc = random.choice(template["descriptions"])
    size = round(random.uniform(*template["size_range"]), 1)
    sol = _sol_counter[mission]
    _file_counter[0] += 1

    base_ci = template.get("chemical_index_base", 0.12)
    sensor_data = {
        "temperature": round(random.gauss(-25 if mission == "mars" else 20 if mission == "satellite" else -20, 8), 2),
        "pressure": round(random.gauss(729 if mission == "mars" else 1013, 8), 2),
        "chemical_index": round(min(1.0, max(0.0, base_ci + random.gauss(0, 0.05))), 3),
        "radiation_level": round(random.uniform(0.1, 0.5), 3),
        "humidity": round(random.uniform(0.0, 0.02), 4),
    }

    if random.random() < 0.08:
        sensor_data["chemical_index"] = round(random.uniform(0.85, 1.0), 3)
        sensor_data["temperature"] = round(sensor_data["temperature"] + random.gauss(0, 25), 2)
        sensor_data["pressure"] = round(sensor_data["pressure"] + random.gauss(0, 30), 2)

    ext = template.get("ext", "dat")
    prefix = template["name_prefix"]
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