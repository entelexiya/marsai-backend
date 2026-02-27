import random
import math
import time
from typing import Dict, Any

# Mars-Earth distance varies from 54.6M to 401M km
# Signal delay: 3 to 22 minutes
# Bandwidth: 0.5 to 6 Mbps depending on orbital geometry

class ChannelSimulator:
    def __init__(self):
        self.mode = "strong"  # strong / medium / weak
        self.bandwidth_mbps = 2.0
        self.packet_loss_percent = 2.0
        self.latency_ms = 800000  # ~13 min in ms
        self.mars_delay_minutes = 13.0
        self.tick = 0
        self._history = []
        self._drift = 0.0

    def update(self) -> Dict[str, Any]:
        self.tick += 1

        # Slow orbital drift — bandwidth changes over hours
        self._drift += random.gauss(0, 0.05)
        self._drift = max(-0.5, min(0.5, self._drift))

        # Occasional sudden drops (solar interference, antenna issues)
        sudden_drop = random.random() < 0.03
        if sudden_drop:
            self.bandwidth_mbps = max(0.1, self.bandwidth_mbps * random.uniform(0.2, 0.5))
        else:
            # Smooth fluctuation
            target_bw = 2.0 + self._drift
            self.bandwidth_mbps += (target_bw - self.bandwidth_mbps) * 0.1
            self.bandwidth_mbps += random.gauss(0, 0.08)
            self.bandwidth_mbps = max(0.1, min(6.0, self.bandwidth_mbps))

        # Packet loss inversely proportional to bandwidth
        self.packet_loss_percent = max(0, 15 - self.bandwidth_mbps * 3) + random.gauss(0, 1)
        self.packet_loss_percent = max(0.0, min(40.0, self.packet_loss_percent))

        # Mars delay varies with distance (simulate orbital motion)
        self.mars_delay_minutes += random.gauss(0, 0.1)
        self.mars_delay_minutes = max(3.0, min(22.0, self.mars_delay_minutes))
        self.latency_ms = self.mars_delay_minutes * 60 * 1000

        # Determine mode
        if self.bandwidth_mbps >= 1.5:
            self.mode = "strong"
        elif self.bandwidth_mbps >= 0.6:
            self.mode = "medium"
        else:
            self.mode = "weak"

        state = self.get_state()
        self._history.append({
            "tick": self.tick,
            "bandwidth_mbps": round(self.bandwidth_mbps, 3),
            "packet_loss_percent": round(self.packet_loss_percent, 2),
        })
        if len(self._history) > 60:
            self._history.pop(0)

        return state

    def get_state(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "bandwidth_mbps": round(self.bandwidth_mbps, 3),
            "packet_loss_percent": round(self.packet_loss_percent, 2),
            "latency_ms": round(self.latency_ms),
            "mars_delay_minutes": round(self.mars_delay_minutes, 1),
            "tick": self.tick,
        }

    def get_history(self):
        return self._history

    def predict_next(self) -> float:
        """Simple trend prediction — is bandwidth going up or down?"""
        if len(self._history) < 3:
            return self.bandwidth_mbps
        recent = [h["bandwidth_mbps"] for h in self._history[-5:]]
        trend = recent[-1] - recent[0]
        predicted = self.bandwidth_mbps + trend * 0.3
        return max(0.1, min(6.0, predicted))

    def set_mars_delay(self, minutes: float):
        self.mars_delay_minutes = max(3.0, min(22.0, minutes))
        self.latency_ms = self.mars_delay_minutes * 60 * 1000
