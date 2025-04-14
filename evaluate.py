"""Evaluate pre-trained LipForensics model on various face forgery datasets"""

import argparse
from collections import defaultdict

import pandas as pd
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop
from tqdm import tqdm

from data.transforms import NormalizeVideo, ToTensorVideo
from data.dataset_clips import ForensicsClips, AVLipsClips, DFDCClips, DemoClips, TestClips
from data.samplers import ConsecutiveClipSampler
from models.spatiotemporal_net import get_model
from utils import get_files_from_split


def parse_args():
    parser = argparse.ArgumentParser(description="DeepFake detector evaluation")
    parser.add_argument(
        "--dataset",
        help="Dataset to evaluate on",
        type=str,
        choices=[
            "FaceForensics++",
            "Deepfakes",
            "FaceSwap",
            "Face2Face",
            "NeuralTextures",
            "FaceShifter",
            "DeeperForensics",
            "CelebDF",
            "DFDC",
            "AVLips",
            "DemoDataset",
            "test",
        ],
        default="FaceForensics++",
    )
    parser.add_argument(
        "--compression",
        help="Video compression level for FaceForensics++",
        type=str,
        choices=["c0", "c23", "c40"],
        default="c23",
    )
    parser.add_argument("--grayscale", dest="grayscale", action="store_true")
    parser.add_argument("--rgb", dest="grayscale", action="store_false")
    parser.set_defaults(grayscale=True)
    parser.add_argument("--frames_per_clip", default=25, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--device", help="Device to put tensors on", type=str, default="cuda:0")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--weights_forgery_path",
        help="Path to pretrained weights for forgery detection",
        type=str,
        default="./models/weights/lipforensics_ff.pth"
    )
    parser.add_argument(
        "--split_path", help="Path to FF++ splits", type=str, default="./data/datasets/Forensics/splits/test.json"
    )
    parser.add_argument(
        "--dfdc_metadata_path", help="Path to DFDC metadata", type=str, default="./data/datasets/DFDC/metadata.json"
    )
    parser.add_argument("--threshold", type=float, default=None)

    args = parser.parse_args()

    return args


def compute_video_level_auc(video_to_logits, video_to_labels, video_to_names, threshold):
    """ "
    Compute video-level area under ROC curve. Averages the logits across the video for non-overlapping clips.

    Parameters
    ----------
    video_to_logits : dict
        Maps video ids to list of logit values
    video_to_labels : dict
        Maps video ids to label
    """
    # Check video_to_logits is not null
    valid_video_ids = [vid for vid in video_to_logits.keys() if len(video_to_logits[vid]) > 0]
    if len(valid_video_ids) == 0:
        raise ValueError("No valid video clips were extracted!")
        
    output_batch = torch.stack([
        torch.mean(torch.stack(video_to_logits[vid]), dim=0) for vid in valid_video_ids
    ])
    output_labels = torch.stack([video_to_labels[vid] for vid in valid_video_ids])
    
    # y_pred, y_true
    y_pred = output_batch.cpu().numpy()
    y_true = output_labels.cpu().numpy()
    
    # AUC
    fpr, tpr, roc_thresholds = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    
    # Confusion Matrix    
    if threshold is None:
        j_scores = tpr - fpr
        best_idx = j_scores.argmax()
        best_threshold = roc_thresholds[best_idx]
    else:
        best_threshold = threshold
    print('best threshold:', best_threshold)
    
    preds = (y_pred > best_threshold).astype(int)
    true_labels = y_true.astype(int)
    cm = metrics.confusion_matrix(true_labels, preds)
    
    records = []
    for vid, score, pred, true in zip(valid_video_ids, y_pred, preds, y_true):
        record = {
            "video_id": vid,
            "score": score,
            "predicted": "fake" if pred == 1 else "real",
            "true": "fake" if true == 1 else "real"
        }
        if video_to_names is not None and vid in video_to_names:
            record["video_name"] = video_to_names[vid]
        records.append(record)
        
    df = pd.DataFrame(records)
    
    return auc, cm, df, best_threshold


def validate_video_level(model, loader, args):
    """ "
    Evaluate model using video-level AUC score.

    Parameters
    ----------
    model : torch.nn.Module
        Model instance
    loader : torch.utils.data.DataLoader
        Loader for forgery data
    args
        Options for evaluation
    """
    model.eval()

    video_to_logits = defaultdict(list)
    video_to_labels = {}
    video_to_names = {}
    with torch.no_grad():
        for data in tqdm(loader):
            images, labels, video_indices, video_name = data
            images = images.to(args.device)
            labels = labels.to(args.device)

            # Forward
            logits = model(images, lengths=[args.frames_per_clip] * images.shape[0])

            # Get maps from video ids to list of logits (representing outputs for clips) as well as to label
            for i in range(len(video_indices)):
                video_id = video_indices[i].item()
                video_to_logits[video_id].append(logits[i])
                video_to_labels[video_id] = labels[i]
                video_to_names[video_id] = video_name[i]

    auc_video, cm, df, threshold = compute_video_level_auc(video_to_logits, video_to_labels, video_to_names, threshold=args.threshold)
    return auc_video, cm, df, threshold


def main():
    args = parse_args()

    model = get_model(weights_forgery_path=args.weights_forgery_path)

    # Get dataset
    transform = Compose(
        [ToTensorVideo(), CenterCrop((88, 88)), NormalizeVideo((0.421,), (0.165,))]
    )
    if args.dataset in [
        "FaceForensics++",
        "Deepfakes",
        "FaceSwap",
        "Face2Face",
        "NeuralTextures",
        "FaceShifter",
        "DeeperForensics",
    ]:
        if args.dataset == "FaceForensics++":
            fake_types = ("Deepfakes", "FaceSwap", "Face2Face", "NeuralTextures")
        else:
            fake_types = (args.dataset,)

        test_split = pd.read_json(args.split_path, dtype=False)
        test_files_real, test_files_fake = get_files_from_split(test_split)

        dataset = ForensicsClips(
            test_files_real,
            test_files_fake,
            args.frames_per_clip,
            grayscale=args.grayscale,
            compression=args.compression,
            fakes=fake_types,
            transform=transform,
            max_frames_per_video=110,
        )
    elif args.dataset == "AVLips":
        dataset = AVLipsClips(args.frames_per_clip, args.grayscale, transform, max_frames_per_video=80)
    elif args.dataset == "DemoDataset":
        dataset = DemoClips(args.frames_per_clip, args.grayscale, transform, max_frames_per_video=100)
    elif args.dataset == "test":
        dataset = TestClips(args.frames_per_clip, args.grayscale, transform, max_frames_per_video=75)
    else:
        metadata = pd.read_json(args.dfdc_metadata_path).T
        dataset = DFDCClips(args.frames_per_clip, metadata, args.grayscale, transform)

    # Get sampler that splits video into non-overlapping clips
    sampler = ConsecutiveClipSampler(dataset.clips_per_video)

    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    
    auc, cm, df, threshold = validate_video_level(model, loader, args)
    print(args.dataset, f"AUC (video-level): {auc}")
    print(f"Confusion Matix: {cm}")
    print(df)
    df.to_csv(f"result_{args.dataset}_gemini_exp4_threshold-{str(threshold)}_100frames.csv", index=False)


if __name__ == "__main__":
    main()
