import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, List
import warnings
warnings.filterwarnings("ignore")

# Reference descriptions of HIGH-VALUE scientific discoveries
# Sentence Transformer will compare file descriptions against these
HIGH_VALUE_REFERENCES = [
    "methane spike detected possible biosignature life on Mars",
    "organic compound carbon molecule biological origin",
    "water ice liquid brine ancient ocean lake river",
    "seismic activity marsquake tectonic interior structure",
    "unusual mineral deposit hematite sulfate phosphate",
    "chemical anomaly unexpected composition spike",
    "atmospheric pressure fluctuation gas release",
    "biosignature habitability life detection",
    "sedimentary rock water deposited ancient environment",
    "meteorite impact crater high energy event",
]

LOW_VALUE_REFERENCES = [
    "routine standard survey no anomalies detected",
    "normal basalt volcanic rock common composition",
    "standard background measurement nominal conditions",
    "dust soil regolith common surface material",
]


class DecisionEngine:
    def __init__(self):
        print("[AI] Loading Sentence Transformer...")
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Pre-encode reference sentences
        self.high_value_embeddings = self.sentence_model.encode(HIGH_VALUE_REFERENCES)
        self.low_value_embeddings = self.sentence_model.encode(LOW_VALUE_REFERENCES)

        print("[AI] Training IsolationForest...")
        self.isolation_forest = self._train_isolation_forest()

        print("[AI] Training LinearRegression for channel prediction...")
        self.channel_predictor = LinearRegression()
        self._channel_history = []
        self._channel_trained = False

        print("[AI] Training RandomForest classifier...")
        self.random_forest = self._train_random_forest()

        self.type_encoder = LabelEncoder()
        self.type_encoder.fit(["IMG", "CHEM", "ATM", "SEISM", "PIXL", "DUST", "LOG"])
        print("[AI] All models ready ✓")

    def _train_isolation_forest(self) -> IsolationForest:
        """Train on 'normal' sensor readings so anomalies stand out."""
        normal_data = []
        for _ in range(2000):
            normal_data.append([
                np.random.normal(-25, 8),    # temperature
                np.random.normal(729, 5),    # pressure
                np.random.uniform(0.0, 0.4), # chemical_index (normal range)
                np.random.uniform(0.1, 0.5), # radiation_level
                np.random.uniform(0.0, 0.02),# humidity
            ])
        model = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
        model.fit(normal_data)
        return model

    def _train_random_forest(self) -> RandomForestClassifier:
        """
        Train on 8000 synthetic examples.
        Features: bandwidth, packet_loss, latency_norm, file_priority,
                  file_size_mb, is_anomaly, semantic_score, predicted_bw,
                  size_bw_ratio, file_type_enc
        Labels: 0=drop, 1=queue, 2=send_now
        """
        X, y = [], []
        for _ in range(8000):
            bw = np.random.uniform(0.1, 6.0)
            loss = np.random.uniform(0, 40)
            latency = np.random.uniform(3, 22)
            priority = np.random.randint(1, 6)
            size = np.random.uniform(0.5, 50)
            is_anomaly = np.random.randint(0, 2)
            sem_score = np.random.uniform(0, 1)
            pred_bw = bw + np.random.normal(0, 0.3)
            size_bw_ratio = size / max(bw, 0.1)
            ftype = np.random.randint(0, 7)

            features = [bw, loss, latency, priority, size, is_anomaly,
                        sem_score, pred_bw, size_bw_ratio, ftype]

            # Decision logic for training labels
            score = (
                priority * 0.3 +
                is_anomaly * 3.0 +
                sem_score * 2.0 +
                (bw / 6.0) * 1.5 -
                (loss / 40.0) * 1.0 -
                (size / 50.0) * 0.5
            )

            if score > 3.5 and bw > 0.5:
                label = 2  # send now
            elif score > 1.5:
                label = 1  # queue
            else:
                label = 0  # drop / wait

            X.append(features)
            y.append(label)

        model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        model.fit(X, y)
        return model

    def analyze_anomaly(self, sensor_data: Dict) -> tuple[bool, float]:
        """IsolationForest: detect sensor anomalies."""
        features = [[
            sensor_data.get("temperature", -25),
            sensor_data.get("pressure", 729),
            sensor_data.get("chemical_index", 0.1),
            sensor_data.get("radiation_level", 0.3),
            sensor_data.get("humidity", 0.01),
        ]]
        score = self.isolation_forest.score_samples(features)[0]
        # score: more negative = more anomalous
        is_anomaly = score < -0.15
        anomaly_strength = max(0.0, min(1.0, (-score - 0.05) / 0.3))
        return is_anomaly, round(anomaly_strength, 3)

    def analyze_semantic(self, description: str) -> float:
        """Sentence Transformer: semantic similarity to high-value references."""
        embedding = self.sentence_model.encode([description])

        # Cosine similarity to high-value references
        high_sims = []
        for ref_emb in self.high_value_embeddings:
            sim = np.dot(embedding[0], ref_emb) / (
                np.linalg.norm(embedding[0]) * np.linalg.norm(ref_emb) + 1e-8
            )
            high_sims.append(sim)

        low_sims = []
        for ref_emb in self.low_value_embeddings:
            sim = np.dot(embedding[0], ref_emb) / (
                np.linalg.norm(embedding[0]) * np.linalg.norm(ref_emb) + 1e-8
            )
            low_sims.append(sim)

        high_score = float(np.max(high_sims))
        low_score = float(np.max(low_sims))

        # Normalize to 0-1
        semantic_score = (high_score - low_score + 1) / 2
        return round(max(0.0, min(1.0, semantic_score)), 3)

    def update_channel_history(self, channel_state: Dict):
        """Feed channel data for LinearRegression prediction."""
        self._channel_history.append(channel_state["bandwidth_mbps"])
        if len(self._channel_history) > 20:
            self._channel_history.pop(0)

        if len(self._channel_history) >= 5:
            X = np.arange(len(self._channel_history)).reshape(-1, 1)
            y = np.array(self._channel_history)
            self.channel_predictor.fit(X, y)
            self._channel_trained = True

    def predict_channel(self) -> float:
        """LinearRegression: predict next bandwidth value."""
        if not self._channel_trained or len(self._channel_history) < 5:
            return self._channel_history[-1] if self._channel_history else 2.0
        next_idx = np.array([[len(self._channel_history)]])
        predicted = self.channel_predictor.predict(next_idx)[0]
        return round(max(0.1, min(6.0, float(predicted))), 3)

    def decide(self, file: Dict, channel_state: Dict) -> Dict[str, Any]:
        """
        Main pipeline:
        1. IsolationForest → anomaly detection
        2. Sentence Transformer → semantic value
        3. LinearRegression → predicted bandwidth
        4. RandomForest → final decision
        """
        # Step 1: Anomaly detection
        is_anomaly, anomaly_score = self.analyze_anomaly(file["sensor_data"])

        # Step 2: Semantic value
        semantic_score = self.analyze_semantic(file["description"])

        # Step 3: Channel prediction
        self.update_channel_history(channel_state)
        predicted_bw = self.predict_channel()
        channel_degrading = predicted_bw < channel_state["bandwidth_mbps"] * 0.7

        # Encode file type
        try:
            ftype_enc = self.type_encoder.transform([file["type"]])[0]
        except Exception:
            ftype_enc = 0

        # Compute priority
        priority = file.get("priority", 1)
        if is_anomaly:
            priority = 5

        size = file.get("size_mb", 10)
        bw = channel_state["bandwidth_mbps"]
        loss = channel_state["packet_loss_percent"]
        latency = channel_state["mars_delay_minutes"]
        size_bw_ratio = size / max(bw, 0.1)

        # Step 4: RandomForest final decision
        features = [[
            bw, loss, latency, priority, size,
            int(is_anomaly), semantic_score,
            predicted_bw, size_bw_ratio, ftype_enc
        ]]
        rf_pred = self.random_forest.predict(features)[0]
        rf_proba = self.random_forest.predict_proba(features)[0]

        # Map prediction to status
        if rf_pred == 2:
            status = "critical" if is_anomaly else "sending"
        elif rf_pred == 1:
            status = "queued"
        else:
            status = "pending"

        # Override: if anomaly AND channel degrading → always send
        if is_anomaly and channel_degrading:
            status = "critical"

        # Build human-readable reason
        reasons = []
        if is_anomaly:
            reasons.append(f"⚡ IsolationForest: ANOMALY detected (score={anomaly_score:.2f})")
        else:
            reasons.append(f"✓ IsolationForest: normal readings")

        if semantic_score > 0.65:
            reasons.append(f"🧠 Semantic value: HIGH ({semantic_score:.2f})")
        elif semantic_score > 0.45:
            reasons.append(f"🧠 Semantic value: MEDIUM ({semantic_score:.2f})")
        else:
            reasons.append(f"🧠 Semantic value: LOW ({semantic_score:.2f})")

        if channel_degrading:
            reasons.append(f"📡 Channel degrading: {bw:.1f}→{predicted_bw:.1f} Mbps — flush critical!")
        else:
            reasons.append(f"📡 Channel stable: {predicted_bw:.1f} Mbps predicted")

        status_label = {"critical": "SEND IMMEDIATELY", "sending": "SEND", "queued": "QUEUE", "pending": "WAIT"}.get(status, "WAIT")
        reasons.append(f"🎯 RandomForest → {status_label} (confidence: {max(rf_proba):.0%})")

        return {
            "status": status,
            "priority": priority,
            "ai_score": round((semantic_score + anomaly_score + (rf_proba[2] if len(rf_proba) > 2 else 0)) / 3, 3),
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": anomaly_score,
            "semantic_score": semantic_score,
            "predicted_bandwidth": predicted_bw,
            "channel_degrading": bool(channel_degrading),
            "decision_reason": " | ".join(reasons),
            "rf_confidence": round(float(max(rf_proba)), 3),
        }
