"""
src/localization/pipeline.py

End-to-End FinFET Drift-Sense Localization Pipeline.
Connects:
1. Input validation & Preprocessing (CLAHE, Bilateral filtering, Gradients)
2. Multi-scale scale-aware search (9:1 to 11:1, rotation ±2°)
3. NMS & Top-K candidate extraction
4. Candidate patch extraction & scale normalization
5. Deep Siamese Network scoring (MobileNetV3 + CIR + CBAM + Dual Correlation)
6. Sub-pixel X/Y coordinate regression offset refinement (dx, dy)
7. Repeated-pattern search-center tie-breaking decision logic
"""

import cv2
import numpy as np
import torch

from src.preprocessing import ImagePreprocessor
from src.candidate import MultiScaleCandidateGenerator, Candidate
from src.siamese import FinFETSiameseNet

class FinFETLocalizer:
    def __init__(self,
                 model_path=None,
                 model_config=None,
                 input_size=128,
                 scale_min=9.0,
                 scale_max=11.0,
                 scale_step=0.25,
                 rotation_min=-2.0,
                 rotation_max=2.0,
                 rotation_step=1.0,
                 top_k=3,
                 nms_radius=40,
                 ambiguity_margin=0.03,
                 device=None):

        self.input_size = input_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.preprocessor = ImagePreprocessor(use_clahe=True, filter_type="bilateral")
        
        self.candidate_generator = MultiScaleCandidateGenerator(
            scale_min=scale_min,
            scale_max=scale_max,
            scale_step=scale_step,
            rotation_min=rotation_min,
            rotation_max=rotation_max,
            rotation_step=rotation_step,
            top_k=top_k,
            nms_radius=nms_radius,
            ambiguity_margin=ambiguity_margin
        )

        # Load Deep Siamese Model if checkpoint provided
        self.model = None
        if model_path is not None and torch.cuda.is_available() or model_path is not None:
            self.model = FinFETSiameseNet(
                in_channels=1,
                use_cbam=model_config.get("use_cbam", True) if model_config else True,
                use_cir=model_config.get("use_cir", True) if model_config else True,
                use_dual_correlation=model_config.get("use_dual_correlation", True) if model_config else True,
                use_xy_feedback=model_config.get("use_xy_feedback", True) if model_config else True
            ).to(self.device)

            if torch.cuda.is_available():
                state_dict = torch.load(model_path, map_location=self.device)
            else:
                state_dict = torch.load(model_path, map_location="cpu")

            self.model.load_state_dict(state_dict)
            self.model.eval()

    def _crop_and_resize(self, img, cx, cy, crop_w, crop_h):
        h, w = img.shape[:2]
        half_w = crop_w / 2.0
        half_h = crop_h / 2.0

        x1 = int(round(cx - half_w))
        y1 = int(round(cy - half_h))
        x2 = int(round(cx + half_w))
        y2 = int(round(cy + half_h))

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
            crop = img_padded[y1 + pad_top:y2 + pad_top, x1 + pad_left:x2 + pad_left]
        else:
            crop = img[y1:y2, x1:x2]

        if crop.shape[0] == 0 or crop.shape[1] == 0:
            crop = np.zeros((int(crop_h), int(crop_w)), dtype=img.dtype)

        return cv2.resize(crop, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)

    def localize(self, ref_img_uint8, search_img_uint8):
        """
        Executes complete localization pipeline given reference and search images.
        
        Returns:
            dict: {
                "x": float,
                "y": float,
                "similarity": float,
                "scale": float,
                "rotation": float,
                "rank": int,
                "candidates": list[dict],
                "selected_candidate": Candidate
            }
        """
        assert ref_img_uint8 is not None, "Reference image is None!"
        assert search_img_uint8 is not None, "Search image is None!"

        # Preprocessing & Gradient Extraction
        ref_enhanced, ref_grads = self.preprocessor.process(ref_img_uint8)
        search_enhanced, search_grads = self.preprocessor.process(search_img_uint8)

        # Multi-scale ZNCC + Gradient Candidate Search & NMS
        candidates = self.candidate_generator.generate_candidates(ref_grads, search_grads)

        if not candidates:
            # Fallback center prediction
            return {
                "x": 500.0,
                "y": 500.0,
                "similarity": 0.0,
                "scale": 10.0,
                "rotation": 0.0,
                "rank": 1,
                "candidates": [],
                "selected_candidate": None
            }

        # Reference patch preparation (1000x1000 -> 128x128)
        ref_patch = cv2.resize(ref_enhanced, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        ref_patch_t = torch.from_numpy(ref_patch.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(self.device)

        scored_candidates = []

        for rank_idx, cand in enumerate(candidates, start=1):
            # Extract search candidate patch around candidate center
            cand_patch = self._crop_and_resize(search_enhanced, cand.x, cand.y, cand.width, cand.height)
            cand_patch_t = torch.from_numpy(cand_patch.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(self.device)

            if self.model is not None:
                with torch.no_grad():
                    sim_t, dx_t, dy_t = self.model(ref_patch_t, cand_patch_t)
                    siamese_sim = float(sim_t.item())
                    dx = float(dx_t.item())
                    dy = float(dy_t.item())
            else:
                siamese_sim = cand.combined_score
                dx, dy = 0.0, 0.0

            # Scale offset adjustments to search image pixel coordinates
            scale_factor = cand.width / self.input_size
            refined_x = cand.x + dx * scale_factor
            refined_y = cand.y + dy * scale_factor

            # Combined score: blend classical ZNCC score + Deep Siamese similarity score
            fused_score = 0.3 * cand.combined_score + 0.7 * siamese_sim

            scored_cand = {
                "x": refined_x,
                "y": refined_y,
                "raw_x": cand.x,
                "raw_y": cand.y,
                "dx": dx,
                "dy": dy,
                "similarity": siamese_sim,
                "zncc_score": cand.combined_score,
                "fused_score": fused_score,
                "scale": cand.scale,
                "rotation": cand.rotation,
                "rank": rank_idx,
                "candidate_obj": cand
            }
            scored_candidates.append(scored_cand)

        # Repeated Pattern decision rule on fused scores
        best_score = max(sc["fused_score"] for sc in scored_candidates)
        ambiguous_sc = [sc for sc in scored_candidates if sc["fused_score"] >= best_score - self.candidate_generator.ambiguity_margin]

        if len(ambiguous_sc) == 1:
            selected = ambiguous_sc[0]
        else:
            # Pick candidate whose center is closest to search image center (500, 500)
            selected = min(ambiguous_sc, key=lambda sc: np.hypot(sc["x"] - 500.0, sc["y"] - 500.0))

        return {
            "x": selected["x"],
            "y": selected["y"],
            "similarity": selected["similarity"],
            "scale": selected["scale"],
            "rotation": selected["rotation"],
            "rank": selected["rank"],
            "candidates": scored_candidates,
            "selected_candidate": selected
        }
