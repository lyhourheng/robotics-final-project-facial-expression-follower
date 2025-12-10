"""
Face Selector Module
Handles target face selection in multi-person scenarios
Supports center-most, largest, and locked tracking modes
"""

import numpy as np
from typing import List, Tuple, Optional


class FaceSelector:
    """
    Select and track a target face from multiple detected faces.
    
    Modes:
    - 'center': Select face closest to image center (default)
    - 'largest': Select largest face (closest person)
    - 'locked': Lock onto a specific face and track it
    """
    
    def __init__(self, mode='center', frame_width=640, frame_height=480):
        """
        Initialize face selector.
        
        Args:
            mode: Selection mode ('center', 'largest', or 'locked')
            frame_width: Camera frame width in pixels
            frame_height: Camera frame height in pixels
        """
        self.mode = mode
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_center_x = frame_width / 2
        self.frame_center_y = frame_height / 2
        
        # Locked tracking state
        self.locked_target = None
        self.lock_history = []
        self.max_lock_lost_frames = 15  # Re-acquire if lost < 15 frames
        self.frames_since_lock_lost = 0
        
    def select_target(self, face_boxes: List[List[int]], 
                     face_scores: List[float]) -> Optional[Tuple[List[int], float, int]]:
        """
        Select target face from detected faces.
        
        Args:
            face_boxes: List of face bounding boxes [[x1, y1, x2, y2], ...]
            face_scores: List of detection confidence scores
            
        Returns:
            Tuple of (target_box, target_score, target_index) or None if no faces
        """
        if len(face_boxes) == 0:
            # No faces detected
            if self.mode == 'locked':
                self.frames_since_lock_lost += 1
                if self.frames_since_lock_lost > self.max_lock_lost_frames:
                    self._reset_lock()
            return None
        
        if self.mode == 'center':
            return self._select_center_most(face_boxes, face_scores)
        elif self.mode == 'largest':
            return self._select_largest(face_boxes, face_scores)
        elif self.mode == 'locked':
            return self._select_locked(face_boxes, face_scores)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _select_center_most(self, face_boxes, face_scores):
        """Select face closest to image center."""
        best_idx = None
        min_distance = float('inf')
        
        for idx, box in enumerate(face_boxes):
            x1, y1, x2, y2 = box
            face_center_x = (x1 + x2) / 2
            face_center_y = (y1 + y2) / 2
            
            # Euclidean distance from frame center
            distance = np.sqrt(
                (face_center_x - self.frame_center_x) ** 2 +
                (face_center_y - self.frame_center_y) ** 2
            )
            
            if distance < min_distance:
                min_distance = distance
                best_idx = idx
        
        if best_idx is not None:
            return face_boxes[best_idx], face_scores[best_idx], best_idx
        return None
    
    def _select_largest(self, face_boxes, face_scores):
        """Select largest face (by area)."""
        best_idx = None
        max_area = 0
        
        for idx, box in enumerate(face_boxes):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            if area > max_area:
                max_area = area
                best_idx = idx
        
        if best_idx is not None:
            return face_boxes[best_idx], face_scores[best_idx], best_idx
        return None
    
    def _select_locked(self, face_boxes, face_scores):
        """
        Track locked target face.
        Re-acquires target if temporarily lost.
        """
        # First detection - lock to center-most face
        if self.locked_target is None:
            result = self._select_center_most(face_boxes, face_scores)
            if result:
                self.locked_target = result[0]  # Store target box
                self.lock_history = [result[0]]
                self.frames_since_lock_lost = 0
            return result
        
        # Target locked - find matching face
        best_match_idx = self._find_matching_face(face_boxes)
        
        if best_match_idx is not None:
            # Target found
            self.locked_target = face_boxes[best_match_idx]
            self.lock_history.append(face_boxes[best_match_idx])
            if len(self.lock_history) > 10:
                self.lock_history.pop(0)
            self.frames_since_lock_lost = 0
            
            return (face_boxes[best_match_idx], 
                   face_scores[best_match_idx], 
                   best_match_idx)
        else:
            # Target lost
            self.frames_since_lock_lost += 1
            
            if self.frames_since_lock_lost > self.max_lock_lost_frames:
                # Lost too long - reset and select new target
                self._reset_lock()
                return self._select_center_most(face_boxes, face_scores)
            else:
                # Still searching - return None but keep lock
                return None
    
    def _find_matching_face(self, face_boxes):
        """
        Find face that matches locked target.
        Uses IoU (Intersection over Union) for matching.
        """
        if self.locked_target is None:
            return None
        
        best_iou = 0
        best_idx = None
        threshold = 0.3  # Minimum IoU to consider a match
        
        for idx, box in enumerate(face_boxes):
            iou = self._calculate_iou(self.locked_target, box)
            
            if iou > best_iou and iou > threshold:
                best_iou = iou
                best_idx = idx
        
        return best_idx
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _reset_lock(self):
        """Reset locked tracking state."""
        self.locked_target = None
        self.lock_history = []
        self.frames_since_lock_lost = 0
    
    def set_mode(self, mode: str):
        """
        Change selection mode.
        
        Args:
            mode: New mode ('center', 'largest', or 'locked')
        """
        if mode not in ['center', 'largest', 'locked']:
            raise ValueError(f"Invalid mode: {mode}")
        
        if mode != self.mode:
            self.mode = mode
            self._reset_lock()  # Reset lock when changing modes
    
    def lock_to_face(self, box: List[int]):
        """
        Manually lock to a specific face box.
        
        Args:
            box: Face bounding box [x1, y1, x2, y2]
        """
        self.mode = 'locked'
        self.locked_target = box
        self.lock_history = [box]
        self.frames_since_lock_lost = 0
    
    def get_target_info(self):
        """Get information about current target."""
        return {
            'mode': self.mode,
            'locked_target': self.locked_target,
            'frames_since_lost': self.frames_since_lock_lost,
            'is_tracking': self.locked_target is not None
        }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("FACE SELECTOR MODULE TEST")
    print("=" * 60)
    
    # Test with sample face detections
    selector = FaceSelector(mode='center', frame_width=640, frame_height=480)
    
    # Simulate 3 detected faces
    face_boxes = [
        [50, 100, 150, 200],    # Left face
        [250, 150, 350, 250],   # Center face (closest to center)
        [450, 100, 550, 200]    # Right face
    ]
    face_scores = [0.75, 0.82, 0.68]
    
    print("\nTest 1: Center-most selection")
    print(f"Frame center: ({selector.frame_center_x}, {selector.frame_center_y})")
    print(f"Detected faces: {len(face_boxes)}")
    
    result = selector.select_target(face_boxes, face_scores)
    if result:
        target_box, target_score, target_idx = result
        print(f"✓ Selected face {target_idx}: box={target_box}, score={target_score:.2f}")
    
    print("\nTest 2: Largest face selection")
    selector.set_mode('largest')
    result = selector.select_target(face_boxes, face_scores)
    if result:
        target_box, target_score, target_idx = result
        print(f"✓ Selected face {target_idx}: box={target_box}, score={target_score:.2f}")
    
    print("\nTest 3: Locked tracking")
    selector.set_mode('locked')
    
    # Frame 1: Lock to center face
    result = selector.select_target(face_boxes, face_scores)
    print(f"Frame 1: Locked to face {result[2] if result else None}")
    
    # Frame 2: Same faces (should track same face)
    result = selector.select_target(face_boxes, face_scores)
    print(f"Frame 2: Tracking face {result[2] if result else None}")
    
    # Frame 3: Target moved slightly
    face_boxes[1] = [260, 160, 360, 260]  # Center face moved
    result = selector.select_target(face_boxes, face_scores)
    print(f"Frame 3: Still tracking face {result[2] if result else None}")
    
    print("\n✓ Face selector tests complete!")
    print("=" * 60)
