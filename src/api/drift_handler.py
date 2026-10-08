from collections import deque

class DriftHandler:
    def __init__(self, base_threshold=0.5, history_size=20, adapt_rate=0.05):
        self.base_threshold = base_threshold
        self.current_threshold = base_threshold
        self.recent_genuine_scores = deque(maxlen=history_size)
        self.adapt_rate = adapt_rate

    def record(self, confidence, decision):
        if decision == "genuine":
            self.recent_genuine_scores.append(confidence)
            self._adapt()

    def _adapt(self):
        if len(self.recent_genuine_scores) < self.recent_genuine_scores.maxlen:
            return

        avg_recent = sum(self.recent_genuine_scores) / len(self.recent_genuine_scores)

        if avg_recent > 0.85:
            target = self.base_threshold - 0.05
        elif avg_recent < 0.55:
            target = self.base_threshold + 0.05
        else:
            target = self.base_threshold

        self.current_threshold += self.adapt_rate * (target - self.current_threshold)
        self.current_threshold = max(0.3, min(0.7, self.current_threshold))

    def get_threshold(self):
        return round(self.current_threshold, 4)