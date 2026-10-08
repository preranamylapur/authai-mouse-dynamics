from collections import deque

class TrustEngine:
    def __init__(self, window_history=5, decay=0.6):
        self.history = deque(maxlen=window_history)
        self.decay = decay

    def update(self, confidence):
        self.history.append(confidence)
        return self.compute_trust_score()

    def compute_trust_score(self):
        if not self.history:
            return 0.5

        weights = [self.decay ** i for i in range(len(self.history))]
        weights.reverse()

        weighted_sum = sum(w * c for w, c in zip(weights, self.history))
        weight_total = sum(weights)

        trust_score = weighted_sum / weight_total
        return round(trust_score, 4)

    def get_status(self, trust_score):
        if trust_score >= 0.75:
            return "trusted"
        elif trust_score >= 0.45:
            return "suspicious"
        else:
            return "untrusted"