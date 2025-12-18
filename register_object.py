#!/usr/bin/env python3
"""
Object Registration Script - Register objects from handheld images.

Usage:
    # Single image
    python register_object.py --image path/to/handheld.jpg --instance_id cellphone1

    # Multiple images (N-shot)
    python register_object.py --image_dir path/to/handheld_dir/ --instance_id cellphone1

    # Custom prompt
    python register_object.py --image_dir path/to/handheld_dir/ --instance_id mug1 --prompt "cup held by hand"

    # Specify output path
    python register_object.py --image path/to/handheld.jpg --instance_id bottle1 --output templates/bottle1

    # Save visualizations
    python register_object.py --image_dir path/to/handheld_dir/ --instance_id cellphone1 --save_viz

    # Register with text attributes from database (for text-based retrieval)
    python register_object.py --image_dir examples/handheld/mug1/ --instance_id mug1 \
        --output templates/my_objects --database examples/object_database.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_loader import ModelLoader
from src.registration.object_registrar import ObjectRegistrar
from src.utils.result_analyzer import draw_recognition_result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Register objects from handheld images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image',
        type=str,
        help='Path to a single handheld image'
    )
    input_group.add_argument(
        '--image_dir',
        type=str,
        help='Directory containing handheld images'
    )

    # Required arguments
    parser.add_argument(
        '--instance_id',
        type=str,
        required=True,
        help='Unique identifier for this object instance'
    )

    # Optional arguments
    parser.add_argument(
        '--output',
        type=str,
        default='templates/objects',
        help='Output path for templates (default: templates/objects)'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='item held by hand',
        help='Detection prompt (default: "item held by hand")'
    )
    parser.add_argument(
        '--save_viz',
        action='store_true',
        help='Save detection and segmentation visualizations'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--database',
        type=str,
        default=None,
        help='Path to object database JSON file for text attributes (default: None)'
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


def load_object_database(database_path: str) -> Dict:
    """Load object database from JSON file."""
    with open(database_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_object_attributes(database: Dict, instance_id: str) -> Optional[List[str]]:
    """Get attributes for an object instance from database."""
    if instance_id in database:
        return database[instance_id].get('attributes', [])
    return None


def save_visualization(image, result, registrar, output_path, image_name, prompt, instance_id):
    """Save visualization of registration result."""
    if not result.success:
        return

    # Re-run detection and segmentation to get visualization data
    det_result = registrar.detector.detect(image, prompt)
    if not det_result['success']:
        return

    bbox = det_result['predictions'][0]
    seg_result = registrar.segmenter.segment_single(image, bbox)

    if seg_result['success']:
        # Format recognition data for visualization
        recognition = {
            'bbox': bbox,
            'instance_id': instance_id,
            'confidence': seg_result['score'],  # Use segmentation score as confidence
            'mask': seg_result['mask'],
            'seg_score': seg_result['score']
        }

        img_with_result = draw_recognition_result(
            image=image,
            recognitions=[recognition],
            confidence_threshold=0.5,
            show_labels=True,
            show_masks=True,
            mask_alpha=0.3
        )

        viz_path = output_path / f"{image_name}.jpg"
        img_with_result.save(viz_path, quality=95)
        print(f"  Saved visualization: {viz_path}")


def load_existing_templates(registrar, output_path):
    """Load existing templates if they exist."""
    templates_file = f"{output_path}_templates.pkl"
    if Path(templates_file).exists():
        try:
            num_loaded = registrar.load_templates(output_path)
            print(f"\nLoaded existing templates: {num_loaded} instance(s)")
            return True
        except Exception as e:
            print(f"\nWarning: Failed to load existing templates: {e}")
            return False
    return False


def main():
    """Main function."""
    args = parse_args()

    print("=" * 80)
    print("Object Registration")
    print("=" * 80)

    # Get image paths
    try:
        image_paths = get_image_paths(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    print(f"\nInstance ID: {args.instance_id}")
    print(f"Images: {len(image_paths)}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Output: {args.output}")

    # Load object database and get attributes if provided
    database = None
    attributes = None
    text_encoder = None

    if args.database:
        print(f"Database: {args.database}")
        try:
            database = load_object_database(args.database)
            attributes = get_object_attributes(database, args.instance_id)
            if attributes:
                print(f"Attributes: {len(attributes)} found for '{args.instance_id}'")
                for attr in attributes:
                    print(f"  - {attr}")
            else:
                print(f"Warning: No attributes found for '{args.instance_id}' in database")
        except FileNotFoundError:
            print(f"Warning: Database file not found: {args.database}")
        except Exception as e:
            print(f"Warning: Failed to load database: {e}")

    # Load models
    print("\n" + "-" * 80)
    print("Loading models...")
    print("-" * 80)

    try:
        loader = ModelLoader(device=args.device, verbose=False)
        # Load text encoder if we have attributes
        include_text_encoder = attributes is not None and len(attributes) > 0
        models = loader.load_all(include_text_encoder=include_text_encoder)
        if include_text_encoder:
            text_encoder = models.get('text_encoder')
        print("✓ Models loaded successfully")
        if text_encoder:
            print(f"✓ Text encoder loaded (dim: {text_encoder.embedding_dim})")
    except Exception as e:
        print(f"✗ Failed to load models: {e}")
        return 1

    # Create registrar
    registrar = ObjectRegistrar(
        detector=models['detector'],
        segmenter=models['segmenter'],
        extractor=models['extractor'],
        default_prompt=args.prompt,
        verbose=False
    )

    # Load existing templates for incremental addition
    output_path = Path(args.output)
    load_existing_templates(registrar, output_path)

    # Prepare visualization directory if needed
    viz_dir = None
    if args.save_viz:
        viz_dir = output_path.parent / f"{output_path.name}_visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

    # Register images
    print("\n" + "-" * 80)
    print("Registering objects...")
    print("-" * 80)

    results = []
    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] {image_path.name}")

        try:
            image = Image.open(image_path).convert('RGB')

            # Use register_with_attributes if attributes are available
            if attributes and text_encoder:
                result = registrar.register_with_attributes(
                    image=image,
                    instance_id=args.instance_id,
                    attributes=attributes,
                    text_encoder=text_encoder,
                    prompt=args.prompt
                )
            else:
                result = registrar.register(
                    image=image,
                    instance_id=args.instance_id,
                    prompt=args.prompt
                )
            results.append(result)

            if result.success:
                print(f"  ✓ Registered successfully")
                print(f"    BBox: {[f'{x:.1f}' for x in result.detection_bbox]}")
                print(f"    Seg score: {result.segmentation_score:.4f}")

                # Save visualization
                if args.save_viz and viz_dir:
                    image_name = f"{args.instance_id}_{i+1:03d}_{image_path.stem}"
                    save_visualization(
                        image, result, registrar, viz_dir,
                        image_name, args.prompt, args.instance_id
                    )
            else:
                print(f"  ✗ Failed: {result.error_message}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append(None)

    # Summary
    print("\n" + "=" * 80)
    print("Registration Summary")
    print("=" * 80)

    num_success = sum(1 for r in results if r and r.success)
    num_total = len(results)

    print(f"Total images processed: {num_total}")
    print(f"Successful: {num_success} ({num_success/num_total*100:.1f}%)")
    print(f"Failed: {num_total - num_success}")

    # Get template statistics
    stats = registrar.get_template_statistics()
    print(f"\nTemplate Statistics:")
    for inst_id, count in stats['templates_per_instance'].items():
        has_text = inst_id in registrar.get_text_embeddings()
        text_marker = " [+text]" if has_text else ""
        print(f"  - {inst_id}: {count} templates{text_marker}")
    print(f"Total instances: {stats['num_instances']}")
    print(f"Total templates: {stats['total_templates']}")
    if stats.get('has_text_embeddings'):
        print(f"Text embeddings: {stats['num_text_embeddings']} instances")

    # Save templates
    print("\n" + "-" * 80)
    print("Saving templates...")
    print("-" * 80)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        templates_path, metadata_path = registrar.save_templates(output_path)
        print(f"✓ Templates: {templates_path}")
        print(f"✓ Metadata: {metadata_path}")
        if args.save_viz and viz_dir:
            print(f"✓ Visualizations: {viz_dir}")
    except Exception as e:
        print(f"✗ Failed to save templates: {e}")
        return 1

    print("=" * 80)

    return 0 if num_success > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
