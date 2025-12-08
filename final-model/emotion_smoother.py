"""
Emotion Smoother Module
Smooth emotion predictions over time to avoid jittery robot behavior
Supports majority vote and exponential moving average (EMA) methods
"""

import numpy as np
from collections import deque, Counter
from typing import List, Optional


class EmotionSmoother:
    """
    Smooth emotion predictions to prevent jittery robot actions.
    
    Methods:
    - 'majority': Simple majority vote over last N frames
    - 'ema': Exponential Moving Average on probability distributions
    """
    
    def __init__(self, method='ema', window_size=5, ema_alpha=0.3, 
                 class_names=['angry', 'happy', 'neutral', 'sad', 'surprised']):
        """
        Initialize emotion smoother.
        
        Args:
            method: Smoothing method ('majority' or 'ema')
            window_size: Number of frames to consider (for majority vote)
            ema_alpha: Smoothing factor for EMA (0-1, lower=smoother)
            class_names: List of emotion class names
        """
        self.method = method
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Majority vote state
        self.emotion_history = deque(maxlen=window_size)
        
        # EMA state
        self.smoothed_probs = None
        
        # Statistics
        self.total_updates = 0
        self.raw_emotions_count = {name: 0 for name in class_names}
        self.smoothed_emotions_count = {name: 0 for name in class_names}
    
    def update(self, emotion: str, confidence: float = None, 
               probabilities: np.ndarray = None) -> str:
        """
        Update smoother with new prediction and return smoothed result.
        
        Args:
            emotion: Predicted emotion class name
            confidence: Prediction confidence (optional)
            probabilities: Full probability distribution (required for EMA)
            
        Returns:
            Smoothed emotion class name
        """
        self.total_updates += 1
        self.raw_emotions_count[emotion] += 1
        
        if self.method == 'majority':
            smoothed = self._update_majority(emotion)
        elif self.method == 'ema':
            if probabilities is None:
                raise ValueError("EMA method requires probabilities array")
            smoothed = self._update_ema(probabilities)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.smoothed_emotions_count[smoothed] += 1
        return smoothed
    
    def _update_majority(self, emotion: str) -> str:
        """
        Majority vote: return most common emotion in recent history.
        """
        self.emotion_history.append(emotion)
        
        if len(self.emotion_history) == 0:
            return emotion
        
        # Count occurrences
        counts = Counter(self.emotion_history)
        most_common = counts.most_common(1)[0][0]
        
        return most_common
    
    def _update_ema(self, probabilities: np.ndarray) -> str:
        """
        Exponential Moving Average: smooth probability distribution.
        """
        # Ensure probabilities is numpy array
        if not isinstance(probabilities, np.ndarray):
            probabilities = np.array(probabilities)
        
        # Initialize on first update
        if self.smoothed_probs is None:
            self.smoothed_probs = probabilities.copy()
        else:
            # EMA formula: smoothed = alpha * new + (1 - alpha) * old
            self.smoothed_probs = (
                self.ema_alpha * probabilities + 
                (1 - self.ema_alpha) * self.smoothed_probs
            )
        
        # Return class with highest smoothed probability
        pred_idx = np.argmax(self.smoothed_probs)
        return self.class_names[pred_idx]
    
    def get_smoothed_probabilities(self) -> Optional[np.ndarray]:
        """Get current smoothed probability distribution (EMA only)."""
        if self.method == 'ema':
            return self.smoothed_probs.copy() if self.smoothed_probs is not None else None
        return None
    
    def get_confidence(self) -> Optional[float]:
        """Get confidence of smoothed prediction (EMA only)."""
        if self.method == 'ema' and self.smoothed_probs is not None:
            return float(np.max(self.smoothed_probs))
        return None
    
    def reset(self):
        """Reset smoother state."""
        self.emotion_history.clear()
        self.smoothed_probs = None
        self.total_updates = 0
        self.raw_emotions_count = {name: 0 for name in self.class_names}
        self.smoothed_emotions_count = {name: 0 for name in self.class_names}
    
    def get_statistics(self) -> dict:
        """Get smoothing statistics."""
        return {
            'method': self.method,
            'total_updates': self.total_updates,
            'window_size': self.window_size if self.method == 'majority' else None,
            'ema_alpha': self.ema_alpha if self.method == 'ema' else None,
            'raw_emotions': self.raw_emotions_count.copy(),
            'smoothed_emotions': self.smoothed_emotions_count.copy(),
            'current_history': list(self.emotion_history) if self.method == 'majority' else None,
            'smoothed_confidence': self.get_confidence()
        }


class AdaptiveEmotionSmoother:
    """
    Advanced smoother with confidence-based adaptive smoothing.
    High confidence predictions get more weight, low confidence get more smoothing.
    """
    
    def __init__(self, base_alpha=0.3, confidence_threshold=0.7,
                 class_names=['angry', 'happy', 'neutral', 'sad', 'surprised']):
        """
        Initialize adaptive smoother.
        
        Args:
            base_alpha: Base EMA alpha for normal confidence
            confidence_threshold: Confidence above which to use higher alpha
            class_names: List of emotion class names
        """
        self.base_alpha = base_alpha
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.smoothed_probs = None
        self.total_updates = 0
    
    def update(self, probabilities: np.ndarray) -> tuple:
        """
        Update with adaptive smoothing based on confidence.
        
        Args:
            probabilities: Probability distribution
            
        Returns:
            Tuple of (emotion_name, smoothed_confidence)
        """
        self.total_updates += 1
        
        # Calculate current confidence
        max_prob = np.max(probabilities)
        
        # Adaptive alpha: high confidence -> higher alpha (less smoothing)
        if max_prob >= self.confidence_threshold:
            alpha = min(0.7, self.base_alpha * 2)  # Trust high confidence
        else:
            alpha = max(0.1, self.base_alpha * 0.5)  # Smooth low confidence more
        
        # Initialize or update smoothed probabilities
        if self.smoothed_probs is None:
            self.smoothed_probs = probabilities.copy()
        else:
            self.smoothed_probs = (
                alpha * probabilities + 
                (1 - alpha) * self.smoothed_probs
            )
        
        # Return result
        pred_idx = np.argmax(self.smoothed_probs)
        emotion = self.class_names[pred_idx]
        confidence = float(self.smoothed_probs[pred_idx])
        
        return emotion, confidence
    
    def reset(self):
        """Reset smoother state."""
        self.smoothed_probs = None
        self.total_updates = 0


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("EMOTION SMOOTHER MODULE TEST")
    print("=" * 60)
    
    class_names = ['angry', 'happy', 'neutral', 'sad', 'surprised']
    
    # Test 1: Majority Vote
    print("\nTest 1: Majority Vote Smoothing")
    print("-" * 60)
    smoother_maj = EmotionSmoother(method='majority', window_size=5, class_names=class_names)
    
    # Simulate noisy predictions
    test_sequence = [
        'happy', 'happy', 'surprised', 'happy', 'happy',  # Brief noise at frame 3
        'neutral', 'happy', 'happy', 'happy', 'happy'     # Brief noise at frame 6
    ]
    
    print("Frame | Raw Emotion  | Smoothed Emotion | Effect")
    print("-" * 60)
    for i, raw_emotion in enumerate(test_sequence, 1):
        smoothed = smoother_maj.update(raw_emotion)
        status = "✓ Stable" if raw_emotion == smoothed else "⚠ Smoothed!"
        print(f"  {i:2d}  | {raw_emotion:12s} | {smoothed:16s} | {status}")
    
    print(f"\nResult: Filtered out {len([e for e in test_sequence if e != 'happy'])} noise frames")
    
    # Test 2: EMA Smoothing
    print("\n" + "=" * 60)
    print("Test 2: Exponential Moving Average (EMA)")
    print("-" * 60)
    smoother_ema = EmotionSmoother(method='ema', ema_alpha=0.3, class_names=class_names)
    
    # Simulate probability distributions
    # Format: [angry, happy, neutral, sad, surprised]
    test_probs = [
        np.array([0.1, 0.8, 0.05, 0.03, 0.02]),  # Happy (high conf)
        np.array([0.1, 0.7, 0.10, 0.05, 0.05]),  # Happy
        np.array([0.2, 0.3, 0.15, 0.10, 0.25]),  # Happy (low conf, noise)
        np.array([0.1, 0.75, 0.05, 0.05, 0.05]), # Happy
        np.array([0.1, 0.8, 0.03, 0.04, 0.03]),  # Happy
    ]
    
    print("Frame | Raw Prediction    | Raw Conf | Smoothed Prediction | Smooth Conf")
    print("-" * 80)
    for i, probs in enumerate(test_probs, 1):
        raw_idx = np.argmax(probs)
        raw_emotion = class_names[raw_idx]
        raw_conf = probs[raw_idx]
        
        smoothed = smoother_ema.update(raw_emotion, probabilities=probs)
        smooth_conf = smoother_ema.get_confidence()
        
        print(f"  {i}   | {raw_emotion:17s} | {raw_conf:.3f}    | "
              f"{smoothed:19s} | {smooth_conf:.3f}")
    
    # Test 3: Adaptive Smoothing
    print("\n" + "=" * 60)
    print("Test 3: Adaptive Smoothing (confidence-based)")
    print("-" * 60)
    smoother_adaptive = AdaptiveEmotionSmoother(base_alpha=0.3, confidence_threshold=0.7, 
                                                 class_names=class_names)
    
    # High confidence -> low confidence -> high confidence
    test_probs_adaptive = [
        np.array([0.05, 0.90, 0.02, 0.02, 0.01]),  # High conf happy
        np.array([0.05, 0.88, 0.03, 0.02, 0.02]),  # High conf happy
        np.array([0.15, 0.40, 0.20, 0.15, 0.10]),  # Low conf (noisy)
        np.array([0.05, 0.85, 0.05, 0.03, 0.02]),  # High conf happy
    ]
    
    print("Frame | Raw Conf | Smoothed Emotion | Smooth Conf | Strategy")
    print("-" * 70)
    for i, probs in enumerate(test_probs_adaptive, 1):
        raw_conf = np.max(probs)
        smoothed, smooth_conf = smoother_adaptive.update(probs)
        strategy = "Trust (high conf)" if raw_conf >= 0.7 else "Smooth (low conf)"
        
        print(f"  {i}   | {raw_conf:.3f}    | {smoothed:16s} | {smooth_conf:.3f}       | {strategy}")
    
    # Statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("-" * 60)
    stats = smoother_maj.get_statistics()
    print(f"Majority Vote - Total updates: {stats['total_updates']}")
    print(f"Raw emotions: {stats['raw_emotions']}")
    print(f"Smoothed emotions: {stats['smoothed_emotions']}")
    
    print("\n✓ Emotion smoother tests complete!")
    print("=" * 60)
    
    print("\nKey Takeaways:")
    print("• Majority vote: Simple, removes brief noise")
    print("• EMA: Smooth transitions, better for gradual changes")
    print("• Adaptive: Trusts high confidence, smooths low confidence")
    print("• Result: Robot has stable, predictable behavior!")
