import random
from typing import Dict, Any
from mission_configs import MISSION_CONFIGS


class ChannelSimulator:
    def __init__(self, mission: str = "mars"):
        self.mission = mission
        self._config = MISSION_CONFIGS[mission]
        self._reset_to_defaults()
        self.tick = 0
        self._history = []
        self._drift = 0.0

    def _reset_to_defaults(self):
        cfg = self._config
        self.bandwidth_mbps = cfg["default_bandwidth"]
        self.delay_seconds = cfg["default_delay"]
        bw_min, bw_max = cfg["bandwidth_range"]
        loss_base = 2.0 if self.mission != "deepspace" else 15.0
        self.packet_loss_percent = loss_base
        self.contact_window = cfg.get("contact_window")
        self.contact_time_remaining = self.contact_window

    def set_mission(self, mission: str):
        self.mission = mission
        self._config = MISSION_CONFIGS[mission]
        self._reset_to_defaults()
        self._history = []
        self._drift = 0.0
        self.tick = 0

    def update(self) -> Dict[str, Any]:
        self.tick += 1
        cfg = self._config
        bw_min, bw_max = cfg["bandwidth_range"]

        self._drift += random.gauss(0, 0.05)
        self._drift = max(-1.0, min(1.0, self._drift))

        # Mission-specific channel behavior
        if self.mission == "satellite":
            # Contact window simulation - satellite passes over ground station
            if self.contact_window:
                if self.contact_time_remaining is not None:
                    self.contact_time_remaining -= 3  # 3 sec per tick
                    if self.contact_time_remaining <= 0:
                        # Out of contact window - next orbit
                        self.bandwidth_mbps = 0.001
                        self.packet_loss_percent = 80.0
                        if random.random() < 0.05:  # new pass
                            self.contact_time_remaining = self.contact_window
                    else:
                        # In contact window - high bandwidth
                        target = bw_max * 0.7 + self._drift * 10
                        self.bandwidth_mbps += (target - self.bandwidth_mbps) * 0.2
                        self.bandwidth_mbps = max(bw_min, min(bw_max, self.bandwidth_mbps))
                        self.packet_loss_percent = max(0, 2 + random.gauss(0, 1))
            else:
                target = (bw_min + bw_max) / 2 + self._drift * 20
                self.bandwidth_mbps += (target - self.bandwidth_mbps) * 0.1
                self.bandwidth_mbps = max(bw_min, min(bw_max, self.bandwidth_mbps))

        elif self.mission == "deepspace":
            # Very slow, mostly stable but DSN scheduling affects it
            sudden_drop = random.random() < 0.02
            if sudden_drop:
                self.bandwidth_mbps *= random.uniform(0.3, 0.6)
            else:
                target = cfg["default_bandwidth"] + self._drift * 0.01
                self.bandwidth_mbps += (target - self.bandwidth_mbps) * 0.05
            self.bandwidth_mbps = max(bw_min, min(bw_max, self.bandwidth_mbps))
            self.packet_loss_percent = max(0, 8 + random.gauss(0, 2))

        else:
            # Mars / Lunar - smooth with occasional drops
            sudden_drop = random.random() < 0.03
            if sudden_drop:
                self.bandwidth_mbps = max(bw_min, self.bandwidth_mbps * random.uniform(0.3, 0.6))
            else:
                target = cfg["default_bandwidth"] + self._drift
                self.bandwidth_mbps += (target - self.bandwidth_mbps) * 0.1
                self.bandwidth_mbps += random.gauss(0, cfg["default_bandwidth"] * 0.04)
            self.bandwidth_mbps = max(bw_min, min(bw_max, self.bandwidth_mbps))
            self.packet_loss_percent = max(0, 8 - self.bandwidth_mbps / bw_max * 8 + random.gauss(0, 1))

        # Delay simulation
        delay_min, delay_max = cfg["delay_range"]
        self.delay_seconds += random.gauss(0, (delay_max - delay_min) * 0.002)
        self.delay_seconds = max(delay_min, min(delay_max, self.delay_seconds))

        # Determine mode
        bw_range = bw_max - bw_min
        if self.bandwidth_mbps >= bw_min + bw_range * 0.6:
            self.mode = "strong"
        elif self.bandwidth_mbps >= bw_min + bw_range * 0.25:
            self.mode = "medium"
        else:
            self.mode = "weak"

        state = self.get_state()
        self._history.append({
            "tick": self.tick,
            "bandwidth_mbps": round(self.bandwidth_mbps, 4),
            "packet_loss_percent": round(self.packet_loss_percent, 2),
        })
        if len(self._history) > 60:
            self._history.pop(0)

        return state

    def get_state(self) -> Dict[str, Any]:
        delay_min, delay_max = self._config["delay_range"]
        return {
            "mode": self.mode,
            "mission": self.mission,
            "bandwidth_mbps": round(self.bandwidth_mbps, 4),
            "packet_loss_percent": round(self.packet_loss_percent, 2),
            "delay_seconds": round(self.delay_seconds, 3),
            # Keep mars_delay_minutes for backwards compat with frontend
            "mars_delay_minutes": round(self.delay_seconds / 60, 3),
            "latency_ms": round(self.delay_seconds * 1000),
            "tick": self.tick,
            "contact_window": self.contact_window,
            "contact_time_remaining": self.contact_time_remaining,
        }

    def get_history(self):
        return self._history

    def predict_next(self) -> float:
        if len(self._history) < 3:
            return self.bandwidth_mbps
        recent = [h["bandwidth_mbps"] for h in self._history[-5:]]
        trend = recent[-1] - recent[0]
        predicted = self.bandwidth_mbps + trend * 0.3
        bw_min, bw_max = self._config["bandwidth_range"]
        return max(bw_min, min(bw_max, predicted))