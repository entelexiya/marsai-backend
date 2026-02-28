import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, Optional
import warnings
import time
warnings.filterwarnings("ignore")

from mission_configs import MISSION_CONFIGS

ALL_FILE_TYPES = ["IMG", "CHEM", "ATM", "SEISM", "PIXL", "SAR", "OPT", "THERMAL",
                  "RADAR", "MULTISPECT", "DRILL", "PLASMA", "MAG", "COSMIC", "PARTICLE", "RADIO"]

# Instrument priority scores — high value instruments per mission
INSTRUMENT_PRIORITY = {
    # Mars — biosignature instruments highest
    "CHEM": 0.9, "PIXL": 0.9, "SEISM": 0.8, "ATM": 0.7, "IMG": 0.5,
    # Satellite — emergency detection highest
    "SAR": 0.85, "THERMAL": 0.8, "OPT": 0.7, "RADAR": 0.65, "MULTISPECT": 0.6,
    # Lunar — resource detection highest
    "DRILL": 0.95, "RADAR": 0.75, "IMG": 0.5,
    # Deep Space — boundary detection highest
    "PLASMA": 0.9, "MAG": 0.85, "COSMIC": 0.9, "PARTICLE": 0.75, "RADIO": 0.7,
}


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

        # Try loading CLIP (optional — graceful fallback if unavailable)
        self.clip_model = None
        self.clip_processor = None
        self._load_clip()

        print("[AI] Training IsolationForest per mission...")
        self._anomaly_thresholds = {}
        self._isolation_forests = {}
        for mission in MISSION_CONFIGS:
            iforest, threshold = self._train_isolation_forest_for_mission(mission)
            self._isolation_forests[mission] = iforest
            self._anomaly_thresholds[mission] = threshold
        # keep legacy aliases for compatibility
        self.isolation_forest = self._isolation_forests["mars"]
        self._anomaly_threshold = self._anomaly_thresholds["mars"]

        print("[AI] Training LinearRegression...")
        self.channel_predictor = LinearRegression()
        self._channel_history = []
        self._channel_trained = False

        print("[AI] Training RandomForest (15 features)...")
        self.random_forest, self.rf_metrics = self._train_random_forest()

        self.type_encoder = LabelEncoder()
        self.type_encoder.fit(ALL_FILE_TYPES)

        # Decision history for metrics dashboard
        self._decision_log = []
        self._anomaly_log = []

        print("[AI] All models ready ✓")
        if self.clip_model:
            print("[AI] CLIP vision model active ✓")
        else:
            print("[AI] CLIP not available — text-only mode")

    def _load_clip(self):
        """Load CLIP model for image analysis — graceful fallback."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            print("[AI] Loading CLIP ViT-B/32...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("[AI] CLIP loaded ✓")
        except Exception as e:
            print(f"[AI] CLIP unavailable: {e}")
            self.clip_model = None
            self.clip_processor = None

    def _train_isolation_forest_for_mission(self, mission: str):
        """Train separate IsolationForest per mission with mission-realistic sensor ranges.
        
        Mars: NASA MEDA real data (PDS archive, Rodriguez-Manfredi et al. 2021)
        Satellite: Earth surface sensor typical ranges (NOAA/ESA reference)
        Lunar: Lunar regolith / vacuum environment (LROC/LRO data ranges)
        Deep Space: Interstellar medium / heliosphere (Voyager in-situ data)
        """
        np.random.seed(42)
        cfg = MISSION_CONFIGS[mission]["sensor_normal"]
        t_mean = cfg["temperature_mean"]
        t_std  = cfg.get("temperature_std", 10)
        p_mean = cfg["pressure_mean"]
        p_std  = cfg.get("pressure_std", 5)
        ci_max = cfg["chemical_index_max"]

        # Mission-specific real data ranges
        MISSION_SENSOR_PARAMS = {
            "mars": {
                # Real NASA MEDA embedded values (PDS, sols 1-847)
                "temp_values": [-23.5,-22.8,-24.1,-23.2,-22.5,-70.2,-72.5,-68.8,-71.4,
                                -55.8,-53.2,-57.1,-54.8,-18.5,-19.2,-17.8,-20.1,-85.2,-88.1],
                "pres_values": [728.4,729.1,731.2,727.8,730.5,728.9,729.7,731.0,728.2,
                                745.2,748.1,751.3,749.8,733.8,732.1,730.5,728.9,718.5,715.2],
                "rad_range": (0.1, 0.5), "hum_range": (0.0, 0.02),
            },
            "satellite": {
                # Earth surface / LEO instrument sensor typical ranges
                "temp_range": (15, 35), "temp_std": 8,
                "pres_range": (980, 1040), "pres_std": 8,
                "rad_range": (0.0, 0.15), "hum_range": (0.0, 0.9),
            },
            "lunar": {
                # Moon: extreme diurnal temp (-170 to +120°C), near-vacuum
                "temp_range": (-170, 120), "temp_std": 60,
                "pres_range": (0, 0.00001), "pres_std": 0,
                "rad_range": (0.2, 0.9), "hum_range": (0.0, 0.0),
            },
            "deepspace": {
                # Deep space: near absolute zero, essentially vacuum
                "temp_range": (-273, -260), "temp_std": 1,
                "pres_range": (0, 0), "pres_std": 0,
                "rad_range": (0.01, 0.3), "hum_range": (0.0, 0.0),
            },
        }

        params = MISSION_SENSOR_PARAMS.get(mission, MISSION_SENSOR_PARAMS["mars"])
        normal_data = []

        for i in range(2000):
            if mission == "mars":
                t_vals = params["temp_values"]
                p_vals = params["pres_values"]
                t = t_vals[i % len(t_vals)] + np.random.normal(0, 3)
                p = p_vals[i % len(p_vals)] + np.random.normal(0, 1.5)
            else:
                t = np.random.uniform(*params["temp_range"]) + np.random.normal(0, params.get("temp_std", 5))
                p = np.random.uniform(*params["pres_range"]) + np.random.normal(0, params.get("pres_std", 1))

            ci  = max(0, min(ci_max, ci_max * 0.5 + np.random.normal(0, ci_max * 0.15)))
            rad = np.random.uniform(*params["rad_range"])
            hum = np.random.uniform(*params["hum_range"])
            normal_data.append([t, p, ci, rad, hum])

        model = IsolationForest(contamination=0.05, random_state=42, n_estimators=200, max_samples=256)
        model.fit(normal_data)
        scores = model.score_samples(normal_data)
        # threshold at 5th percentile of training scores
        threshold = float(np.percentile(scores, 5))
        # store score range for normalization
        self._score_ranges = getattr(self, '_score_ranges', {})
        self._score_ranges[mission] = {
            "min": float(np.percentile(scores, 1)),
            "max": float(np.percentile(scores, 99)),
        }
        print(f"[AI] IsolationForest [{mission}]: threshold={threshold:.4f}, score_range=[{self._score_ranges[mission]['min']:.3f}, {self._score_ranges[mission]['max']:.3f}]")
        return model, threshold

    def _train_random_forest(self):
        """Train RandomForest with 15 features including instrument priority, sol age, data ratio."""
        X, y = [], []
        np.random.seed(42)

        for _ in range(10000):
            bw = np.random.uniform(0.001, 150.0)
            bw_norm = np.log1p(bw) / np.log1p(150)
            loss = np.random.uniform(0, 40)
            latency = np.random.uniform(0, 480)
            priority = np.random.randint(1, 6)
            size = np.random.uniform(0.05, 500)
            is_anomaly = np.random.randint(0, 2)
            sem_score = np.random.uniform(0, 1)
            pred_bw_norm = np.clip(bw_norm + np.random.normal(0, 0.05), 0, 1)
            size_bw_ratio = min(size / max(bw, 0.001), 100)
            ftype = np.random.randint(0, len(ALL_FILE_TYPES))

            # NEW features
            instrument_priority = np.random.uniform(0.3, 1.0)   # instrument scientific value
            sol_age = np.random.uniform(0, 100)                  # sols since collection
            sol_age_norm = 1.0 - min(sol_age / 100, 1.0)        # fresher = more urgent
            data_ratio = np.random.uniform(1, 20)                # collected/transmittable ratio
            clip_score = np.random.uniform(0, 1)                 # CLIP image score (0 if not image)
            channel_volatility = np.random.uniform(0, 1)         # channel stability

            features = [
                bw_norm, loss, latency, priority, size,
                is_anomaly, sem_score, pred_bw_norm, size_bw_ratio, ftype,
                instrument_priority, sol_age_norm, data_ratio, clip_score, channel_volatility
            ]

            score = (
                priority * 0.25 +
                is_anomaly * 3.0 +
                sem_score * 2.0 +
                instrument_priority * 1.5 +
                bw_norm * 1.2 +
                sol_age_norm * 0.8 +
                clip_score * 1.0 -
                (loss / 40.0) * 0.8 -
                min(size / 500, 1) * 0.4 -
                channel_volatility * 0.3
            )

            if score > 4.0 and bw_norm > 0.05:
                label = 2
            elif score > 2.0:
                label = 1
            else:
                label = 0

            X.append(features)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        # Train/test split for metrics
        split = int(len(X) * 0.85)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Compute real metrics
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
            "train_samples": int(split),
            "test_samples": int(len(X_test)),
            "n_features": 15,
            "feature_names": [
                "bandwidth_norm", "packet_loss", "latency", "priority", "file_size",
                "is_anomaly", "semantic_score", "pred_bw_norm", "size_bw_ratio", "file_type",
                "instrument_priority", "sol_age_norm", "data_ratio", "clip_score", "channel_volatility"
            ],
            "feature_importance": {
                name: round(float(imp), 4)
                for name, imp in zip(
                    ["bandwidth_norm", "packet_loss", "latency", "priority", "file_size",
                     "is_anomaly", "semantic_score", "pred_bw_norm", "size_bw_ratio", "file_type",
                     "instrument_priority", "sol_age_norm", "data_ratio", "clip_score", "channel_volatility"],
                    model.feature_importances_
                )
            }
        }

        print(f"[AI] RandomForest: accuracy={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} (15 features, {split} samples)")
        return model, metrics

    def analyze_anomaly(self, sensor_data: Dict, mission: str = "mars") -> tuple:
        """
        Detect anomalies using mission-specific IsolationForest.
        Each mission has its own model trained on realistic sensor ranges.
        Score is normalized relative to that mission's training distribution.
        """
        model = self._isolation_forests.get(mission, self._isolation_forests["mars"])
        threshold = self._anomaly_thresholds.get(mission, self._anomaly_thresholds["mars"])
        score_range = getattr(self, '_score_ranges', {}).get(mission, {"min": -0.6, "max": -0.05})

        features = [[
            sensor_data.get("temperature", -25),
            sensor_data.get("pressure", 729),
            sensor_data.get("chemical_index", 0.1),
            sensor_data.get("radiation_level", 0.3),
            sensor_data.get("humidity", 0.01),
        ]]
        score = float(model.score_samples(features)[0])
        is_anomaly = score < threshold

        # Normalize anomaly strength properly using mission's score distribution
        s_min = score_range["min"]   # most anomalous seen in training
        s_max = score_range["max"]   # most normal seen in training
        # anomaly_strength: 0 = very normal, 1 = very anomalous
        anomaly_strength = max(0.0, min(1.0, (s_max - score) / max(s_max - s_min, 0.01)))
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

    def analyze_image_clip(self, image_url: str = None, image_bytes: bytes = None, mission: str = "mars") -> float:
        """
        Analyze image using CLIP — returns scientific value score 0-1.
        Compares image against mission-specific text prompts.
        """
        if self.clip_model is None:
            return 0.5  # fallback

        try:
            from PIL import Image
            import requests
            import torch
            import io as _io

            # Load image
            if image_bytes:
                image = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
            elif image_url:
                resp = requests.get(image_url, timeout=8)
                image = Image.open(_io.BytesIO(resp.content)).convert("RGB")
            else:
                return 0.5

            # Mission-specific CLIP prompts
            high_value_prompts = {
                "mars": [
                    "unusual mineral deposit on Mars surface",
                    "rock formation with layered sedimentary structure",
                    "chemical anomaly biosignature Mars rover",
                    "water ice polar cap Mars",
                    "methane plume atmospheric anomaly",
                ],
                "satellite": [
                    "flood disaster emergency aerial view",
                    "wildfire smoke satellite image",
                    "deforestation land change satellite",
                    "oil spill ocean pollution",
                ],
                "lunar": [
                    "water ice lunar south pole crater",
                    "lava tube entrance moon surface",
                    "regolith drilling sample moon",
                ],
                "deepspace": [
                    "plasma nebula interstellar medium",
                    "cosmic ray burst space",
                    "magnetic field visualization space",
                ],
            }
            low_value_prompts = [
                "routine normal surface scan",
                "standard background measurement",
                "flat featureless terrain",
            ]

            high_prompts = high_value_prompts.get(mission, high_value_prompts["mars"])
            all_prompts = high_prompts + low_value_prompts

            inputs = self.clip_processor(
                text=all_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = logits.softmax(dim=0).numpy()

            high_score = float(np.sum(probs[:len(high_prompts)]))
            return round(min(1.0, high_score * 1.2), 3)

        except Exception as e:
            print(f"[CLIP] Error: {e}")
            return 0.5

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
        """Enhanced: EMA + trend + volatility blending + LinearRegression sanity check."""
        if len(self._channel_history) < 3:
            return self._channel_history[-1] if self._channel_history else 2.0

        history = np.array(self._channel_history)
        weights = np.exp(np.linspace(-1, 0, len(history)))
        weights /= weights.sum()
        ema = float(np.dot(weights, history))

        if len(history) >= 5:
            recent = history[-5:]
            trend = float(np.polyfit(np.arange(5), recent, 1)[0])
            predicted = ema + trend * 2
        else:
            predicted = ema

        volatility = float(np.std(history[-10:])) if len(history) >= 10 else 0
        mean_bw = float(np.mean(history[-20:])) if len(history) >= 20 else ema
        blend = min(volatility / (bw_max * 0.1 + 1e-6), 0.7)
        predicted = predicted * (1 - blend) + mean_bw * blend

        if self._channel_trained and len(history) >= 5:
            next_idx = np.array([[len(history)]])
            lr_pred = float(self.channel_predictor.predict(next_idx)[0])
            predicted = (predicted * 0.6 + lr_pred * 0.4)

        return round(max(bw_min, min(bw_max, predicted)), 4)

    def decide(self, file: Dict, channel_state: Dict, clip_score: Optional[float] = None) -> Dict[str, Any]:
        mission = file.get("mission", channel_state.get("mission", "mars"))
        cfg = MISSION_CONFIGS.get(mission, MISSION_CONFIGS["mars"])
        bw_min, bw_max = cfg["bandwidth_range"]

        is_anomaly, anomaly_score = self.analyze_anomaly(file["sensor_data"], mission)
        semantic_score = self.analyze_semantic(file["description"], mission)

        bw = channel_state["bandwidth_mbps"]
        self.update_channel_history(bw)
        predicted_bw = self.predict_channel(bw_min, bw_max)
        channel_degrading = predicted_bw < bw * 0.7

        # Channel volatility
        if len(self._channel_history) >= 5:
            channel_volatility = float(np.std(self._channel_history[-10:])) / (bw_max + 1e-6)
        else:
            channel_volatility = 0.1

        try:
            ftype_enc = int(self.type_encoder.transform([file["type"]])[0])
        except Exception:
            ftype_enc = 0

        # Instrument priority
        instrument_priority = INSTRUMENT_PRIORITY.get(file.get("type", "IMG"), 0.5)

        # Sol age (fresher data = higher urgency)
        sol_age = float(file.get("sol_age", 0))
        sol_age_norm = 1.0 - min(sol_age / 100.0, 1.0)

        # Data ratio for this mission
        data_ratio = min(cfg.get("data_ratio", 10), 20)

        # CLIP score — use provided or 0.5 for non-image files
        if clip_score is None:
            clip_score = file.get("clip_score", 0.5 if file.get("type") in ["IMG", "OPT", "SAR"] else 0.3)

        priority = int(file.get("priority", 1))
        if is_anomaly:
            priority = 5

        bw_norm = float(np.log1p(bw) / np.log1p(150))
        pred_bw_norm = float(np.log1p(predicted_bw) / np.log1p(150))
        size = float(file.get("size_mb", 10))
        loss = float(channel_state["packet_loss_percent"])
        latency = float(channel_state.get("delay_seconds", channel_state.get("mars_delay_minutes", 13) * 60) / 60)
        size_bw_ratio = min(size / max(bw, 0.001), 100)

        features = [[
            bw_norm, loss, latency, priority, size,
            int(is_anomaly), semantic_score, pred_bw_norm, size_bw_ratio, ftype_enc,
            instrument_priority, sol_age_norm, data_ratio, float(clip_score), channel_volatility
        ]]

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

        # Build decision reasons
        reasons = []
        if is_anomaly:
            reasons.append(f"⚡ IsolationForest: ANOMALY (score={anomaly_score:.2f})")
        else:
            reasons.append(f"✓ IsolationForest: normal (NASA MEDA baseline)")

        if semantic_score > 0.65:
            reasons.append(f"🧠 Semantic [{mission}]: HIGH ({semantic_score:.2f})")
        elif semantic_score > 0.45:
            reasons.append(f"🧠 Semantic [{mission}]: MEDIUM ({semantic_score:.2f})")
        else:
            reasons.append(f"🧠 Semantic [{mission}]: LOW ({semantic_score:.2f})")

        if file.get("type") in ["IMG", "OPT", "SAR"] and self.clip_model:
            reasons.append(f"👁️ CLIP vision: {clip_score:.2f}")

        if channel_degrading:
            reasons.append(f"📡 Degrading: {bw:.4f}→{predicted_bw:.4f} Mbps — flush!")
        else:
            reasons.append(f"📡 EMA+LR: {predicted_bw:.4f} Mbps predicted")

        reasons.append(f"🔬 Instrument [{file.get('type','?')}]: priority={instrument_priority:.2f}")

        label = {"critical": "SEND NOW", "sending": "SEND", "queued": "QUEUE", "pending": "WAIT"}.get(status, "WAIT")
        reasons.append(f"🎯 RandomForest (15 feat) → {label} ({max(rf_proba):.0%})")

        result = {
            "status": status,
            "priority": int(priority),
            "ai_score": float(round((semantic_score + anomaly_score + float(rf_proba[2] if len(rf_proba) > 2 else 0)) / 3, 3)),
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": float(anomaly_score),
            "semantic_score": float(semantic_score),
            "clip_score": float(clip_score),
            "instrument_priority": float(instrument_priority),
            "predicted_bandwidth": float(predicted_bw),
            "channel_degrading": bool(channel_degrading),
            "decision_reason": " | ".join(reasons),
            "rf_confidence": float(round(float(max(rf_proba)), 3)),
            "mission": mission,
            "clip_active": self.clip_model is not None,
        }

        # Log for metrics
        self._decision_log.append({
            "ts": time.time(),
            "status": status,
            "mission": mission,
            "semantic_score": float(semantic_score),
            "is_anomaly": bool(is_anomaly),
            "rf_confidence": float(max(rf_proba)),
        })
        if len(self._decision_log) > 200:
            self._decision_log.pop(0)

        return result

    def get_metrics(self) -> Dict:
        """Return model performance metrics for dashboard."""
        recent = self._decision_log[-100:] if self._decision_log else []
        status_counts = {"critical": 0, "sending": 0, "queued": 0, "pending": 0}
        for d in recent:
            status_counts[d.get("status", "pending")] = status_counts.get(d.get("status", "pending"), 0) + 1

        avg_confidence = float(np.mean([d["rf_confidence"] for d in recent])) if recent else 0
        avg_semantic = float(np.mean([d["semantic_score"] for d in recent])) if recent else 0
        anomaly_rate = float(sum(1 for d in recent if d["is_anomaly"]) / max(len(recent), 1))

        return {
            "rf_metrics": self.rf_metrics,
            "recent_decisions": len(recent),
            "status_distribution": status_counts,
            "avg_rf_confidence": round(avg_confidence, 3),
            "avg_semantic_score": round(avg_semantic, 3),
            "anomaly_rate": round(anomaly_rate, 3),
            "clip_active": self.clip_model is not None,
            "models": {
                "isolation_forest": "NASA MEDA data (PDS archive)",
                "sentence_transformer": "all-MiniLM-L6-v2 (pretrained)",
                "random_forest": f"15 features, {self.rf_metrics['train_samples']} samples",
                "channel_predictor": "EMA + LinearRegression ensemble",
                "clip": "openai/clip-vit-base-patch32" if self.clip_model else "not loaded",
            }
        }