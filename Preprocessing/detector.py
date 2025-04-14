import time
import json
import os
import pandas as pd
from PIL import Image
from preprocessing2 import VisualPreprocess  # Module for video processing


def process_multiple_videos(video_target, video_paths, output_dir, processor, label):
    """
    Process multiple videos to extract mouth frames.

    Parameters:
        video_target (dict): Target process timing for each video（start_time, end_time）
        video_paths (list): Input video paths
        output_dir (str): Output directory for processed videos
        processor (VisualPreprocess): Mouth detection processor
        label (str): safe/scam label for the videos

    Returns:
        dict: including video paths, labels, mouth frame counts, and processing times
    """
    proc_dict = {}
    for video_path in video_paths:
        video_name = os.path.basename(video_path)
        try:
            start_sec = video_target[video_name]["start_time"]
            end_sec = video_target[video_name]["end_time"]
            if start_sec is not None or end_sec is not None:
                print(f"Processing {video_name} from {start_sec}s to {end_sec}s")
                each_start_time = time.time()

                mouth_frames, cnt = processor.process_video(
                    video_path,
                    start_time=int(start_sec) + 1,
                    end_time=int(end_sec) - 1,
                )

                if mouth_frames:
                    video_output_path = os.path.join(
                        output_dir, os.path.splitext(video_name)[0]
                    )
                    os.makedirs(video_output_path, exist_ok=True)
                    for key, img in mouth_frames.items():
                        Image.fromarray(img).save(
                            os.path.join(video_output_path, f"{key}.jpg")
                        )
                else:
                    print(f"No mouth frames found in: {video_path}")

                each_proc_time = time.time() - each_start_time
                print(f"Time taken: {each_proc_time:.2f} seconds\n{'-'*40}")

                proc_dict.setdefault("video", []).append(video_path)
                proc_dict.setdefault("label", []).append(label)
                proc_dict.setdefault("mouths_cnt", []).append(cnt)
                proc_dict.setdefault("proc_time", []).append(each_proc_time)

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
    return proc_dict


if __name__ == "__main__":
    all_start_time = time.time()
    proc_dict = {}

    # Initialize the processor
    processor = VisualPreprocess(
        model_path="./preprocessing/face_detection_yunet_2023mar.onnx",
        mean_face="./preprocessing/20words_mean_face.npy",
        use_cuda=True,
        num_workers=4,
    )

    # Setting input video paths and targets
    safe_video_path = "./data/datasets/DemoDataset/videos/safe_videos"
    scam_video_path = "./data/datasets/DemoDataset/videos/scam_videos"
    safe_target = json.load(
        open("./data/datasets/DemoDataset/result_by_safe_video_exp.json")
    )
    scam_target = json.load(
        open("./data/datasets/DemoDataset/result_by_scam_video_exp.json")
    )

    # Setting output directories
    safe_output = (
        "./data/datasets/DemoDataset/cropped_mouths_orig_gemini_exp/safe_videos"
    )
    scam_output = (
        "./data/datasets/DemoDataset/cropped_mouths_orig_gemini_exp/scam_videos"
    )
    os.makedirs(safe_output, exist_ok=True)
    os.makedirs(scam_output, exist_ok=True)

    # Process safe videos
    safe_paths = [os.path.join(safe_video_path, v) for v in os.listdir(safe_video_path)]
    temp_result = process_multiple_videos(
        safe_target, safe_paths, safe_output, processor, label="safe"
    )
    for key in temp_result:
        proc_dict.setdefault(key, []).append(temp_result[key])

    # Process scam videos
    scam_paths = [os.path.join(scam_video_path, v) for v in os.listdir(scam_video_path)]
    temp_result = process_multiple_videos(
        scam_target, scam_paths, scam_output, processor, label="scam"
    )
    for key in temp_result:
        proc_dict.setdefault(key, []).append(temp_result[key])

    # Save the processing results
    all_proc_time = time.time() - all_start_time
    print(f"All processing completed in {all_proc_time:.2f} seconds")

    proc_df = pd.DataFrame.from_dict(proc_dict)
    proc_df.to_csv("DemoDataset_gemini_preprocess_exp4_info.csv", index=False)
    print(proc_df.head())
