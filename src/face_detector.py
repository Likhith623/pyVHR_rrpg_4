"""
Face detection and ROI extraction using MediaPipe Face Mesh.
Extracts forehead and cheek ROIs for rPPG signal computation.
"""
import cv2
import numpy as np
import mediapipe as mp


class FaceROIExtractor:
    """Detects face landmarks and extracts skin ROI regions."""

    # MediaPipe Face Mesh landmark indices for each ROI
    FOREHEAD_INDICES = [10, 67, 109, 108, 69, 104, 68, 71, 63, 105,
                        66, 107, 9, 336, 296, 334, 293, 301, 298,
                        338, 337, 299, 297, 332]
    LEFT_CHEEK_INDICES = [187, 123, 116, 117, 118, 119, 120, 121,
                          128, 245, 193, 55, 65, 52, 53]
    RIGHT_CHEEK_INDICES = [411, 352, 345, 346, 347, 348, 349, 350,
                           357, 465, 417, 285, 295, 282, 283]

    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def extract_roi_means(self, frame):
        """
        Extract mean RGB from forehead, left cheek, right cheek.

        Returns:
            dict with keys 'forehead', 'left_cheek', 'right_cheek',
            each a 3-element array [R, G, B], or None if no face found.
            Also returns 'face_bbox' as (x, y, w, h).
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        points = np.array(
            [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]
        )

        roi_data = {}
        for name, indices in [
            ("forehead", self.FOREHEAD_INDICES),
            ("left_cheek", self.LEFT_CHEEK_INDICES),
            ("right_cheek", self.RIGHT_CHEEK_INDICES),
        ]:
            roi_pts = points[indices]
            mask = np.zeros((h, w), dtype=np.uint8)
            hull = cv2.convexHull(roi_pts)
            cv2.fillConvexPoly(mask, hull, 255)
            mean_val = cv2.mean(frame, mask=mask)[:3]  # BGR
            roi_data[name] = np.array([mean_val[2], mean_val[1], mean_val[0]])  # RGB

        # Compute face bounding box
        all_x = points[:, 0]
        all_y = points[:, 1]
        bbox = (all_x.min(), all_y.min(),
                all_x.max() - all_x.min(), all_y.max() - all_y.min())
        roi_data["face_bbox"] = bbox
        roi_data["landmarks"] = points

        return roi_data

    def release(self):
        self.face_mesh.close()
