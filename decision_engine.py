import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from typing import Dict, Any
import warnings
warnings.filterwarnings("ignore")

from mission_configs import MISSION_CONFIGS

ALL_FILE_TYPES = ["IMG", "CHEM", "ATM", "SEISM", "PIXL", "SAR", "OPT", "THERMAL",
                  "RADAR", "MULTISPECT", "DRILL", "PLASMA", "MAG", "COSMIC", "PARTICLE", "RADIO"]


class DecisionEngine:
    def __init__(self):
        print("[AI] Loading Sentence Transformer...")
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Pre-encode references for ALL missions
        self._mission_embeddings = {}
        for mission, cfg in MISSION_CONFIGS.items():
            self._mission_embeddings[mission] = {
                "high": self.sentence_model.encode(cfg["high_value_refs"]),
                "low": self.sentence_model.encode(cfg["low_value_refs"]),
            }

        print("[AI] Training IsolationForest...")
        self.isolation_forest = self._train_isolation_forest()

        print("[AI] Training LinearRegression...")
        self.channel_predictor = LinearRegression()
        self._channel_history = []
        self._channel_trained = False

        print("[AI] Training RandomForest...")
        self.random_forest = self._train_random_forest()

        self.type_encoder = LabelEncoder()
        self.type_encoder.fit(ALL_FILE_TYPES)
        print("[AI] All models ready ✓")

    def _train_isolation_forest(self) -> IsolationForest:
        normal_data = []
        for _ in range(3000):
            normal_data.append([
                np.random.normal(-25, 8),
                np.random.normal(729, 5),
                np.random.uniform(0.0, 0.30),
                np.random.uniform(0.1, 0.5),
                np.random.uniform(0.0, 0.02),
            ])
        for _ in range(500):
            normal_data.append([
                np.random.normal(-25, 15),
                np.random.normal(729, 10),
                np.random.uniform(0.25, 0.55),
                np.random.uniform(0.3, 0.7),
                np.random.uniform(0.01, 0.04),
            ])
        model = IsolationForest(contamination=0.05, random_state=42, n_estimators=200, max_samples=0.8)
        model.fit(normal_data)
        return model

    def _train_random_forest(self) -> RandomForestClassifier:
        X, y = [], []
        for _ in range(8000):
            bw = np.random.uniform(0.001, 150.0)
            bw_norm = np.log1p(bw) / np.log1p(150)
            loss = np.random.uniform(0, 40)
            latency = np.random.uniform(0, 480)
            priority = np.random.randint(1, 6)
            size = np.random.uniform(0.05, 500)
            is_anomaly = np.random.randint(0, 2)
            sem_score = np.random.uniform(0, 1)
            pred_bw_norm = bw_norm + np.random.normal(0, 0.05)
            size_bw_ratio = min(size / max(bw, 0.001), 100)
            ftype = np.random.randint(0, len(ALL_FILE_TYPES))

            features = [bw_norm, loss, latency, priority, size, is_anomaly,
                        sem_score, pred_bw_norm, size_bw_ratio, ftype]

            score = (
                priority * 0.3 + is_anomaly * 3.0 + sem_score * 2.0 +
                bw_norm * 1.5 - (loss / 40.0) * 1.0 - min(size / 500, 1) * 0.5
            )

            if score > 3.5 and bw_norm > 0.1:
                label = 2
            elif score > 1.5:
                label = 1
            else:
                label = 0

            X.append(features)
            y.append(label)

        model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        model.fit(X, y)
        return model

    def analyze_anomaly(self, sensor_data: Dict) -> tuple:
        features = [[
            sensor_data.get("temperature", -25),
            sensor_data.get("pressure", 729),
            sensor_data.get("chemical_index", 0.1),
            sensor_data.get("radiation_level", 0.3),
            sensor_data.get("humidity", 0.01),
        ]]
        score = self.isolation_forest.score_samples(features)[0]
        is_anomaly = score < -0.35
        anomaly_strength = max(0.0, min(1.0, (-score - 0.05) / 0.3))
        return is_anomaly, round(float(anomaly_strength), 3)

    def analyze_semantic(self, description: str, mission: str = "mars") -> float:
        embedding = self.sentence_model.encode([description])
        embs = self._mission_embeddings.get(mission, self._mission_embeddings["mars"])

        high_sims = [
            float(np.dot(embedding[0], ref) / (np.linalg.norm(embedding[0]) * np.linalg.norm(ref) + 1e-8))
            for ref in embs["high"]
        ]
        low_sims = [
            float(np.dot(embedding[0], ref) / (np.linalg.norm(embedding[0]) * np.linalg.norm(ref) + 1e-8))
            for ref in embs["low"]
        ]

        semantic_score = (max(high_sims) - max(low_sims) + 1) / 2
        return round(max(0.0, min(1.0, semantic_score)), 3)

    def update_channel_history(self, bw: float):
        self._channel_history.append(bw)
        if len(self._channel_history) > 20:
            self._channel_history.pop(0)
        if len(self._channel_history) >= 5:
            X = np.arange(len(self._channel_history)).reshape(-1, 1)
            y = np.array(self._channel_history)
            self.channel_predictor.fit(X, y)
            self._channel_trained = True

    def predict_channel(self, bw_min: float = 0.1, bw_max: float = 6.0) -> float:
        if not self._channel_trained or len(self._channel_history) < 5:
            return self._channel_history[-1] if self._channel_history else 2.0
        next_idx = np.array([[len(self._channel_history)]])
        predicted = self.channel_predictor.predict(next_idx)[0]
        return round(max(bw_min, min(bw_max, float(predicted))), 4)

    def decide(self, file: Dict, channel_state: Dict) -> Dict[str, Any]:
        mission = file.get("mission", channel_state.get("mission", "mars"))
        cfg = MISSION_CONFIGS.get(mission, MISSION_CONFIGS["mars"])
        bw_min, bw_max = cfg["bandwidth_range"]

        is_anomaly, anomaly_score = self.analyze_anomaly(file["sensor_data"])
        semantic_score = self.analyze_semantic(file["description"], mission)

        bw = channel_state["bandwidth_mbps"]
        self.update_channel_history(bw)
        predicted_bw = self.predict_channel(bw_min, bw_max)
        channel_degrading = predicted_bw < bw * 0.7

        try:
            ftype_enc = int(self.type_encoder.transform([file["type"]])[0])
        except Exception:
            ftype_enc = 0

        priority = int(file.get("priority", 1))
        if is_anomaly:
            priority = 5

        bw_norm = float(np.log1p(bw) / np.log1p(150))
        pred_bw_norm = float(np.log1p(predicted_bw) / np.log1p(150))
        size = float(file.get("size_mb", 10))
        loss = float(channel_state["packet_loss_percent"])
        latency = float(channel_state.get("delay_seconds", channel_state.get("mars_delay_minutes", 13) * 60) / 60)
        size_bw_ratio = min(size / max(bw, 0.001), 100)

        features = [[bw_norm, loss, latency, priority, size, int(is_anomaly),
                     semantic_score, pred_bw_norm, size_bw_ratio, ftype_enc]]
        rf_pred = self.random_forest.predict(features)[0]
        rf_proba = self.random_forest.predict_proba(features)[0]

        if rf_pred == 2:
            status = "critical" if is_anomaly else "sending"
        elif rf_pred == 1:
            status = "queued"
        else:
            status = "pending"

        if is_anomaly and channel_degrading:
            status = "critical"

        reasons = []
        if is_anomaly:
            reasons.append(f"⚡ IsolationForest: ANOMALY (score={anomaly_score:.2f})")
        else:
            reasons.append(f"✓ IsolationForest: normal")

        if semantic_score > 0.65:
            reasons.append(f"🧠 Semantic [{mission}]: HIGH ({semantic_score:.2f})")
        elif semantic_score > 0.45:
            reasons.append(f"🧠 Semantic [{mission}]: MEDIUM ({semantic_score:.2f})")
        else:
            reasons.append(f"🧠 Semantic [{mission}]: LOW ({semantic_score:.2f})")

        if channel_degrading:
            reasons.append(f"📡 Degrading: {bw:.3f}→{predicted_bw:.3f} Mbps — flush!")
        else:
            reasons.append(f"📡 Stable: {predicted_bw:.3f} Mbps predicted")

        label = {"critical": "SEND NOW", "sending": "SEND", "queued": "QUEUE", "pending": "WAIT"}.get(status, "WAIT")
        reasons.append(f"🎯 RandomForest → {label} ({max(rf_proba):.0%})")

        return {
            "status": status,
            "priority": int(priority),
            "ai_score": float(round((semantic_score + anomaly_score + (float(rf_proba[2]) if len(rf_proba) > 2 else 0)) / 3, 3)),
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": float(anomaly_score),
            "semantic_score": float(semantic_score),
            "predicted_bandwidth": float(predicted_bw),
            "channel_degrading": bool(channel_degrading),
            "decision_reason": " | ".join(reasons),
            "rf_confidence": float(round(float(max(rf_proba)), 3)),
            "mission": mission,
        }