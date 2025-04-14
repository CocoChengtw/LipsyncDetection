import ray
import cv2
import numpy as np
import torch
from collections import deque
import face_alignment
from skimage import transform as tf
from yunet import YuNet
import time
from PIL import Image


# Align the face landmarks to a standard size
def warp_img(src, dst, img, std_size):
    """
    Align the face landmarks to a standard size
    """
    tform = tf.estimate_transform("similarity", src, dst)
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)
    return (warped * 255).astype("uint8"), tform


def apply_transform(transform, img, std_size):
    """
    Transform the image using the given transformation
    """
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    return (warped * 255).astype("uint8")


# Cut a patch from the image based on landmarks
def cut_patch(img, landmarks, height, width, threshold=5):
    """
    Cut a patch from the image based on landmarks
    """
    center_x, center_y = np.mean(landmarks, axis=0)
    if center_y - height < 0:
        center_y = height
    if int(center_y) - height < 0 - threshold:
        raise Exception("too much bias in height")

    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception("too much bias in width")

    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception("too much bias in height")

    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception("too much bias in width")

    return np.copy(
        img[
            int(round(center_y) - round(height)) : int(round(center_y) + round(height)),
            int(round(center_x) - round(width)) : int(round(center_x) + round(width)),
        ]
    )


# Ray remote class for parallel processing
@ray.remote(num_gpus=1)
class FrameProcessor:
    def __init__(self, model_path, mean_face_landmarks, device="cuda:0"):
        # Initialize YuNet and face alignment models
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, flip_input=False, device=device
        )
        self.yunet = self._initialize_yunet(model_path)
        self.mean_face_landmarks = mean_face_landmarks

        # Set parameters for processing
        self.STD_SIZE = (256, 256)
        self.STABLE_POINTS = [33, 36, 39, 42, 45]
        self.MOUTH_START_IDX = 48
        self.MOUTH_STOP_IDX = 68
        self.CROP_HEIGHT = 96
        self.CROP_WIDTH = 96

    def _initialize_yunet(self, model_path):
        # Initialize YuNet model
        return YuNet(
            modelPath=model_path,
            inputSize=[320, 320],
            confThreshold=0.4,
            nmsThreshold=0.4,
            topK=1,
            backendId=0,
            targetId=0,
        )

    def process_frame(self, name, frame):
        # Extract face landmarks and crop mouth region
        h, w, _ = frame.shape
        self.yunet.setInputSize([w, h])
        results = self.yunet.infer(frame)

        if not len(results):
            return name, None, None

        bbox = results[0][0:4].astype(np.int32)
        X, Y, W, H = bbox[0], bbox[1], bbox[2], bbox[3]
        W, H = X + W, Y + H
        X, Y = max(0, X), max(0, Y)
        W, H = min(w, W), min(h, H)

        face = frame[Y:H, X:W]
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        try:
            preds = self.fa.get_landmarks(rgb_face)
            if not preds:
                return name, None, None
            return name, rgb_face, preds[0]
        except Exception as e:
            print(f"Error in landmark detection for {name}: {str(e)}")
            return name, None, None


# Main class for video preprocessing
class VisualPreprocess:
    def __init__(self, model_path, mean_face, use_cuda=True, num_workers=4):
        # Initiate Ray
        if ray.is_initialized():
            ray.shutdown()
        available_gpus = torch.cuda.device_count() if use_cuda else 0
        ray.init(num_cpus=num_workers, num_gpus=available_gpus)

        self.device = "cuda" if use_cuda else "cpu"
        self.mean_face_landmarks = np.load(mean_face)
        self.window_margin = 12
        self.num_workers = num_workers

        self.workers = [
            FrameProcessor.remote(model_path, self.mean_face_landmarks, self.device)
            for _ in range(num_workers)
        ]

        self.STD_SIZE = (256, 256)
        self.STABLE_POINTS = [33, 36, 39, 42, 45]
        self.MOUTH_START_IDX = 48
        self.MOUTH_STOP_IDX = 68
        self.CROP_HEIGHT = 96
        self.CROP_WIDTH = 96

    def extract_frames(self, video_path, start_time, end_time):
        # Extract frames from video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = {}
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count >= (end_frame - start_frame):
                break
            key = f'{video_path.split("/")[-1].replace(".mp4", "").replace(".MP4", "")}_{frame_count:04d}'
            frames[key] = frame
            frame_count += 1

        cap.release()
        return frames

    def process_frame_sequence(self, frames):
        # Process a sequence of frames using multiple workers in parellel
        futures = []
        for name, frame in frames.items():
            worker_idx = hash(name) % self.num_workers
            futures.append(self.workers[worker_idx].process_frame.remote(name, frame))

        q_frames, q_landmarks, q_names = deque(), deque(), deque()
        transform = None
        mouths = {}

        results = ray.get(futures)
        for result in results:
            if result is None:
                continue
            name, frame, landmark = result
            if frame is None or landmark is None:
                continue
            q_frames.append(frame)
            q_landmarks.append(landmark)
            q_names.append(name)

            if len(q_frames) == self.window_margin:
                try:
                    mouths, transform = self._process_window(
                        q_frames, q_landmarks, q_names, mouths
                    )
                except Exception as e:
                    print(f"Error in processing window: {str(e)}")
                    q_frames.clear()
                    q_landmarks.clear()
                    q_names.clear()

        # Transform remaining frames using the last transform
        if transform is not None and len(q_frames) > 0:
            try:
                mouths = self._process_remaining(
                    q_frames, q_landmarks, q_names, transform, mouths
                )
            except Exception as e:
                print(f"Error in processing remaining frames: {str(e)}")

        return mouths

    def _process_window(self, q_frames, q_landmarks, q_name, mouths):
        # Smooth the landmarks and process the window
        smoothed_landmarks = np.mean(q_landmarks, axis=0)
        cur_landmarks = q_landmarks.popleft()
        cur_frame = q_frames.popleft()
        cur_name = q_name.popleft()

        trans_frame, transform = warp_img(
            smoothed_landmarks[self.STABLE_POINTS, :],
            self.mean_face_landmarks[self.STABLE_POINTS, :],
            cur_frame,
            self.STD_SIZE,
        )

        trans_landmarks = transform(cur_landmarks)
        cropped_frame = cut_patch(
            trans_frame,
            trans_landmarks[self.MOUTH_START_IDX : self.MOUTH_STOP_IDX],
            self.CROP_HEIGHT // 2,
            self.CROP_WIDTH // 2,
        )

        mouths[cur_name] = cropped_frame.astype(np.uint8)
        return mouths, transform

    def _process_remaining(self, q_frames, q_landmarks, q_name, transform, mouths):
        # Use previous window transform to process remaining frames
        while q_frames:
            cur_frame = q_frames.popleft()
            cur_name = q_name.popleft()
            cur_landmarks = q_landmarks.popleft()

            trans_frame = apply_transform(transform, cur_frame, self.STD_SIZE)
            trans_landmarks = transform(cur_landmarks)
            cropped_frame = cut_patch(
                trans_frame,
                trans_landmarks[self.MOUTH_START_IDX : self.MOUTH_STOP_IDX],
                self.CROP_HEIGHT // 2,
                self.CROP_WIDTH // 2,
            )
            mouths[cur_name] = cropped_frame.astype(np.uint8)
        return mouths

    def process_video(self, video_path, start_time=0, end_time=3):
        # Main Function to process video
        try:
            print("video:", video_path)
            start_process_time = time.time()

            frames = self.extract_frames(video_path, start_time, end_time)
            if frames is None or len(frames) == 0:
                print(f"No frames extracted from video: {video_path}")
                return None, 0

            mouths = self.process_frame_sequence(frames)
            if mouths is None or len(mouths) == 0:
                print(f"No mouth regions detected in video: {video_path}")
                return None, 0

            process_time = time.time() - start_process_time
            print(f"Processing completed in {process_time:.2f} seconds")
            print(f"Processed {len(mouths)} frames")

            return mouths, len(mouths)

        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            return None, 0
