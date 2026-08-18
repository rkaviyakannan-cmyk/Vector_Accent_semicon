"""
src/candidate/search.py

Multi-scale scale-aware candidate generator using ZNCC (Zero-mean Normalized Cross-Correlation)
on intensity and gradient representations, with Non-Maximum Suppression (NMS) and 
the mandatory Repeated-Pattern Search-Centre Tie-Breaking Rule.
"""

import cv2
import numpy as np

class Candidate:
    def __init__(self, x, y, width, height, scale, rotation, intensity_score, gradient_score, combined_score):
        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.height = float(height)
        self.scale = float(scale)
        self.rotation = float(rotation)
        self.intensity_score = float(intensity_score)
        self.gradient_score = float(gradient_score)
        self.combined_score = float(combined_score)

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "scale": self.scale,
            "rotation": self.rotation,
            "intensity_score": self.intensity_score,
            "gradient_score": self.gradient_score,
            "combined_score": self.combined_score
        }

    def __repr__(self):
        return (f"Candidate(x={self.x:.1f}, y={self.y:.1f}, scale={self.scale:.2f}, "
                f"rot={self.rotation:.1f}deg, score={self.combined_score:.4f})")


class MultiScaleCandidateGenerator:
    def __init__(self,
                 scale_min=9.0,
                 scale_max=11.0,
                 scale_step=0.25,
                 rotation_min=-2.0,
                 rotation_max=2.0,
                 rotation_step=1.0,
                 top_k=3,
                 nms_radius=40,
                 alpha=0.5,
                 beta=0.5,
                 ambiguity_margin=0.03,
                 search_center=(500.0, 500.0)):
        
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_step = scale_step
        self.rotation_min = rotation_min
        self.rotation_max = rotation_max
        self.rotation_step = rotation_step
        self.top_k = top_k
        self.nms_radius = nms_radius
        self.alpha = alpha
        self.beta = beta
        self.ambiguity_margin = ambiguity_margin
        self.search_center = search_center

    def _rotate_image(self, img, angle_deg):
        """Rotates image around its center."""
        if abs(angle_deg) < 1e-3:
            return img
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def generate_candidates(self, ref_grads, search_grads):
        """
        Generates candidate locations across scales and rotations.
        
        Parameters:
            ref_grads (dict): Dictionary with 'intensity' and 'gradient_magnitude' arrays for reference image (1000x1000).
            search_grads (dict): Dictionary with 'intensity' and 'gradient_magnitude' arrays for search image (1000x1000).
            
        Returns:
            list[Candidate]: Top K candidates sorted by score and center ambiguity rule.
        """
        ref_int = ref_grads["intensity"]
        ref_grad = ref_grads["gradient_magnitude"]
        
        src_int = search_grads["intensity"]
        src_grad = search_grads["gradient_magnitude"]
        
        scales = np.arange(self.scale_min, self.scale_max + 1e-5, self.scale_step)
        rotations = np.arange(self.rotation_min, self.rotation_max + 1e-5, self.rotation_step)
        
        all_candidates = []
        
        for scale in scales:
            # Target footprint in search image
            t_w = int(round(ref_int.shape[1] / scale))
            t_h = int(round(ref_int.shape[0] / scale))
            if t_w < 10 or t_h < 10:
                continue
                
            for rot in rotations:
                # Rotate reference template
                rot_ref_int = self._rotate_image(ref_int, rot)
                rot_ref_grad = self._rotate_image(ref_grad, rot)
                
                # Resize template to target footprint scale
                tmpl_int = cv2.resize(rot_ref_int, (t_w, t_h), interpolation=cv2.INTER_AREA)
                tmpl_grad = cv2.resize(rot_ref_grad, (t_w, t_h), interpolation=cv2.INTER_AREA)
                
                # Match template using ZNCC (CV_32F input)
                res_int = cv2.matchTemplate(src_int, tmpl_int, cv2.TM_CCOEFF_NORMED)
                res_grad = cv2.matchTemplate(src_grad, tmpl_grad, cv2.TM_CCOEFF_NORMED)
                
                # Combined score map
                res_comb = self.alpha * res_int + self.beta * res_grad
                
                # Extract local maxima peaks
                h_map, w_map = res_comb.shape
                # Flatten indices to find top peaks per scale/rotation
                flat = res_comb.flatten()
                top_indices = np.argpartition(flat, -self.top_k)[-self.top_k:]
                
                for idx in top_indices:
                    top_y, top_x = np.unravel_index(idx, (h_map, w_map))
                    score_comb = res_comb[top_y, top_x]
                    score_int = res_int[top_y, top_x]
                    score_grad = res_grad[top_y, top_x]
                    
                    # Convert top-left (top_x, top_y) to candidate center (cx, cy)
                    cx = top_x + t_w / 2.0
                    cy = top_y + t_h / 2.0
                    
                    cand = Candidate(
                        x=cx,
                        y=cy,
                        width=t_w,
                        height=t_h,
                        scale=scale,
                        rotation=rot,
                        intensity_score=score_int,
                        gradient_score=score_grad,
                        combined_score=score_comb
                    )
                    all_candidates.append(cand)

        # Apply Non-Maximum Suppression (NMS) across all scales/rotations
        nms_candidates = self._apply_nms(all_candidates)
        
        # Select top K candidates
        top_k_candidates = nms_candidates[:self.top_k]
        return top_k_candidates

    def _apply_nms(self, candidates):
        """Applies NMS based on spatial distance threshold."""
        if not candidates:
            return []
            
        # Sort candidates by combined score descending
        sorted_cands = sorted(candidates, key=lambda c: c.combined_score, reverse=True)
        keep = []
        
        for cand in sorted_cands:
            overlap = False
            for k in keep:
                dist = np.hypot(cand.x - k.x, cand.y - k.y)
                if dist < self.nms_radius:
                    overlap = True
                    break
            if not overlap:
                keep.append(cand)
                
        return keep

    def select_final_candidate(self, candidates):
        """
        Applies the mandatory Repeated-Pattern Rule:
        If top candidates have scores within `ambiguity_margin` of the best score,
        pick the candidate whose center is closest to (500, 500).
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
            
        # Sort by combined score descending
        best_score = max(c.combined_score for c in candidates)
        
        # Filter ambiguous candidates
        ambiguous = [c for c in candidates if c.combined_score >= best_score - self.ambiguity_margin]
        
        if len(ambiguous) == 1:
            return ambiguous[0]
            
        # Tie-break using distance to search image center (500, 500)
        cx0, cy0 = self.search_center
        selected = min(ambiguous, key=lambda c: np.hypot(c.x - cx0, c.y - cy0))
        return selected
