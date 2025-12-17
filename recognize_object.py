#!/usr/bin/env python3
"""
Object Recognition Script - Recognize registered objects in query scenes.

Usage:
    # Single image
    python recognize_object.py --image path/to/query.jpg --templates templates/objects

    # Multiple images
    python recognize_object.py --image_dir path/to/query_scenes/ --templates templates/objects

    # Adjust confidence threshold
    python recognize_object.py --image path/to/query.jpg --templates templates/objects --threshold 0.6

    # Specify output directory
    python recognize_object.py --image_dir path/to/query_scenes/ --templates templates/objects --output recognition_results/
"""

import argparse
import sys
from pathlib import Path
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_loader import ModelLoader
from src.recognition.object_recognizer import ObjectRecognizer
from src.utils.result_analyzer import draw_recognition_result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Recognize registered objects in query scenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image',
        type=str,
        help='Path to a single query image'
    )
    input_group.add_argument(
        '--image_dir',
        type=str,
        help='Directory containing query images'
    )

    # Required arguments
    parser.add_argument(
        '--templates',
        type=str,
        required=True,
        help='Path to template files (without extension)'
    )

    # Optional arguments
    parser.add_argument(
        '--output',
        type=str,
        default='recognition_results',
        help='Output directory for results (default: recognition_results)'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='objects',
        help='Detection prompt (default: "objects")'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Confidence threshold (default: 0.5)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )

    return parser.parse_args()


def get_image_paths(args):
    """Get list of image paths from arguments."""
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return [image_path]

    elif args.image_dir:
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {image_dir}")

        # Get all image files
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(sorted(image_dir.glob(ext)))

        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")

        return image_paths

    return []


def filter_best_matches(recognitions, threshold):
    """Keep only the highest confidence match per instance ID."""
    best_matches = {}

    for rec in recognitions:
        if rec['confidence'] < threshold:
            continue

        inst_id = rec['instance_id']

        if inst_id not in best_matches or rec['confidence'] > best_matches[inst_id]['confidence']:
            best_matches[inst_id] = rec

    return list(best_matches.values())


def save_visualization(image, recognitions, output_dir, image_name, threshold):
    """Save visualization of recognition results."""
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    img_with_results = draw_recognition_result(
        image=image,
        recognitions=recognitions,
        confidence_threshold=threshold,
        show_labels=True,
        show_masks=True,
        mask_alpha=0.3
    )

    save_path = viz_dir / f"{image_name}_recognition.jpg"
    img_with_results.save(save_path, quality=95)

    return save_path


def main():
    """Main function."""
    args = parse_args()

    print("=" * 80)
    print("Object Recognition")
    print("=" * 80)

    # Get image paths
    try:
        image_paths = get_image_paths(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    print(f"\nQuery images: {len(image_paths)}")
    print(f"Templates: {args.templates}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {args.output}")

    # Load models
    print("\n" + "-" * 80)
    print("Loading models...")
    print("-" * 80)

    try:
        loader = ModelLoader(device=args.device, verbose=False)
        models = loader.load_all()
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load models: {e}")
        return 1

    # Create recognizer
    recognizer = ObjectRecognizer(
        detector=models['detector'],
        segmenter=models['segmenter'],
        extractor=models['extractor'],
        matcher=models['matcher'],
        default_prompt=args.prompt,
        confidence_threshold=args.threshold,
        verbose=False
    )

    # Load templates
    print("\n" + "-" * 80)
    print("Loading templates...")
    print("-" * 80)

    try:
        num_loaded = recognizer.load_templates(args.templates)
        print(f"✓ Loaded {num_loaded} instance(s)")

        template_info = recognizer.get_template_info()
        for inst_id in template_info['instance_ids']:
            count = template_info['templates_per_instance'][inst_id]
            print(f"  - {inst_id}: {count} templates")
    except Exception as e:
        print(f"✗ Failed to load templates: {e}")
        return 1

    # Prepare output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Recognize objects
    print("\n" + "-" * 80)
    print("Recognizing objects...")
    print("-" * 80)

    all_results = []
    total_detections = 0
    total_matches = 0

    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] {image_path.name}")

        try:
            image = Image.open(image_path).convert('RGB')
            result = recognizer.recognize(
                image=image,
                prompt=args.prompt,
                return_features=False,
                return_masks=True
            )

            if result.success:
                # Filter to best matches
                best_matches = filter_best_matches(
                    result.recognitions,
                    args.threshold
                )

                print(f"  Detections: {result.num_detections}")
                print(f"  Matches (>={args.threshold}): {len(best_matches)}")

                if best_matches:
                    print(f"  Recognized objects:")
                    for j, rec in enumerate(best_matches, 1):
                        inst_id = rec['instance_id']
                        conf = rec['confidence']
                        print(f"    {j}. {inst_id} (confidence: {conf:.4f})")

                        # Show top-3 alternative matches
                        if 'top_k_matches' in rec and len(rec['top_k_matches']) > 1:
                            print(f"       Top-3 matches:")
                            for rank, (match_id, score) in enumerate(rec['top_k_matches'][:3], 1):
                                print(f"         {rank}. {match_id}: {score:.4f}")

                # Save visualization
                viz_path = save_visualization(
                    image, best_matches, output_dir,
                    image_path.stem, args.threshold
                )
                print(f"  ✓ Saved: {viz_path}")

                total_detections += result.num_detections
                total_matches += len(best_matches)

                all_results.append({
                    'image': image_path.name,
                    'detections': result.num_detections,
                    'matches': len(best_matches)
                })
            else:
                print(f"  ✗ Failed: {result.error_message}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Final summary
    print("\n" + "=" * 80)
    print("Recognition Summary")
    print("=" * 80)

    print(f"Total images: {len(image_paths)}")
    print(f"Total detections: {total_detections}")
    print(f"Total matches (>={args.threshold}): {total_matches}")
    if image_paths:
        print(f"Avg detections per image: {total_detections/len(image_paths):.2f}")
        print(f"Avg matches per image: {total_matches/len(image_paths):.2f}")

    print(f"\nResults saved to: {output_dir}")
    print(f"  - Visualizations: {output_dir}/visualizations/")

    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
