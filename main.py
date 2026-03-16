"""
Neuro-Pulse: rPPG-Based Liveness & Deepfake Detection System

Two use cases:
  1. Webcam Liveness Detection (threshold-based, real-time)
  2. Video Deepfake Detection (ML/DL classifier on 35 rPPG features)
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Neuro-Pulse: rPPG-Based Liveness & Deepfake Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real-time webcam liveness detection
  python main.py liveness

  # Extract features from dataset for ML training
  python main.py extract --real-dir /path/to/real --fake-dir /path/to/fake

  # Extract features using CHROM method
  python main.py extract --method CHROM
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ─── Liveness Detection ───────────────────────────────────────
    live_parser = subparsers.add_parser(
        "liveness", help="Real-time webcam liveness detection (Use Case 1)"
    )
    live_parser.add_argument("--camera", type=int, default=0, help="Camera index")

    # ─── Feature Extraction ───────────────────────────────────────
    extract_parser = subparsers.add_parser(
        "extract", help="Extract rPPG features from video dataset (Use Case 2)"
    )
    extract_parser.add_argument("--real-dir", type=str, default=None,
                                help="Directory of real videos")
    extract_parser.add_argument("--fake-dir", type=str, default=None,
                                help="Directory of deepfake videos")
    extract_parser.add_argument("--output", type=str, default="./output",
                                help="Output directory for features")
    extract_parser.add_argument("--method", type=str, default="GREEN",
                                choices=["GREEN", "CHROM", "POS"],
                                help="rPPG extraction method")
    extract_parser.add_argument("--max-frames", type=int, default=300,
                                help="Max frames per video (300 = ~10s at 30fps)")

    args = parser.parse_args()

    if args.command == "liveness":
        from src.liveness_detector import run_liveness_detection
        run_liveness_detection()

    elif args.command == "extract":
        from src.video_pipeline import process_dataset
        process_dataset(
            real_dir=args.real_dir,
            fake_dir=args.fake_dir,
            method=args.method,
            max_frames=args.max_frames,
            output_dir=args.output,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
