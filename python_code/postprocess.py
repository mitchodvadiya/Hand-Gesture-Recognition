from collections import Counter
from config import Config

class HGRPostProcessor:
    """
    Implements the post-processing filter algorithm from the paper:
    'Assessing the influence of LSTM and post-processing in CNN based HGR using EMG'
    """
    def __init__(self, mode: str = Config.POSTPROCESS_MODE):
        self.mode = mode

    @staticmethod
    def classify_predictions(labels: list) -> str:
        """
        Classifies a sequence of predicted frame labels into a single gesture.
        1. Filters out 'noGesture' predictions.
        2. Counts the occurrences of each gesture.
        3. If max count >= MIN_LABELS_SEQUENCE (default 4), outputs the dominant gesture.
           Otherwise outputs 'noGesture'.
        """
        active_labels = [l for l in labels if l != "noGesture"]
        if not active_labels:
            return "noGesture"
            
        counts = Counter(active_labels)
        most_common_class, top_count = counts.most_common(1)[0]
        
        if top_count >= Config.MIN_LABELS_SEQUENCE:
            return most_common_class
        return "noGesture"

    def filter_labels(self, labels: list, dominant_class: str) -> list:
        """
        Post-processing sequence filtering algorithm (1-1, 1-2, 2-1).
        Removes spurious transient predictions and fills false-negative gaps.
        """
        if not labels or dominant_class == "noGesture":
            return labels
            
        filtered = list(labels)
        n = len(filtered)
        if n < 3:
            return filtered
            
        mode = self.mode

        # 1. Check the first label
        right = (filtered[1] == "noGesture")
        if mode == '1-2' and n > 2:
            right = (filtered[1] == dominant_class) or (filtered[2] == "noGesture")
        current = (filtered[0] == dominant_class)
        if right and current:
            filtered[0] = "noGesture"

        # 2. Middle labels range
        start = 2 if mode == '2-1' and n > 3 else 1
        finish = n - 2 if mode == '1-2' and n > 3 else n - 1

        for i in range(start, finish):
            # Left check
            if mode == '2-1' and i >= 2:
                left = (filtered[i - 1] == dominant_class) or (filtered[i - 2] == dominant_class)
            else:
                left = (filtered[i - 1] == dominant_class)

            # Right check
            if mode == '1-2' and i + 2 < n:
                right = (filtered[i + 1] == dominant_class) or (filtered[i + 2] == dominant_class)
            else:
                right = (filtered[i + 1] == dominant_class)

            current_is_target = (filtered[i] == dominant_class)

            # Rule A: Fill single gap inside a gesture sequence
            if left and right and not current_is_target:
                filtered[i] = dominant_class

            # Rule B: Remove single spurious gesture spike in a noGesture region
            if (not left) and (not right) and current_is_target:
                filtered[i] = "noGesture"

        # 3. Check the last label
        left = (filtered[n - 2] == "noGesture")
        if mode == '2-1' and n > 2:
            left = (filtered[n - 2] == "noGesture") or (filtered[n - 3] == "noGesture")
        current = (filtered[-1] == dominant_class)
        if left and current:
            filtered[-1] = "noGesture"

        # 4. Clean up any remaining non-dominant gesture labels
        for i in range(n):
            if filtered[i] != dominant_class:
                filtered[i] = "noGesture"

        return filtered

    def process(self, predicted_labels: list, apply_filter: bool = True) -> str:
        """
        Full evaluation pipeline for a sample sequence:
        1. Find preliminary dominant class
        2. Optionally apply sequence filtering
        3. Final classification
        """
        if not apply_filter:
            return self.classify_predictions(predicted_labels)
            
        initial_class = self.classify_predictions(predicted_labels)
        if initial_class == "noGesture":
            return "noGesture"
            
        cleaned_labels = self.filter_labels(predicted_labels, initial_class)
        return self.classify_predictions(cleaned_labels)
