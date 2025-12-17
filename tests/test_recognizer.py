#!/usr/bin/env python3
"""
Test script for ObjectRecognizer functionality.

This script tests the object recognition pipeline including:
- Loading registered templates
- Detecting and recognizing objects in query scenes
- Visualization of recognition results with confidence scores
- Generating recognition reports
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_loader import ModelLoader
from src.registration.object_registrar import ObjectRegistrar
from src.recognition.object_recognizer import ObjectRecognizer
from src.utils.result_analyzer import RecognitionVisualizer


def setup_output_directory(base_dir: str = "recognition_output") -> Path:
    """
    Setup output directory structure for recognition results.

    Args:
        base_dir: Base directory name

    Returns:
        Path to output directory
    """
    output_dir = PROJECT_ROOT / base_dir
    output_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (output_dir / "visualizations").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"  - visualizations/: Recognition result images")
    print(f"  - reports/: Recognition reports (TXT and JSON)\n")

    return output_dir


def register_templates_if_needed(
    models: dict,
    templates_path: Path,
    handheld_dir: Path
) -> bool:
    """
    Register templates if they don't already exist.

    Args:
        models: Dictionary of loaded models
        templates_path: Path where templates should be saved
        handheld_dir: Directory containing handheld images

    Returns:
        True if templates were created/loaded successfully
    """
    templates_file = f"{templates_path}_templates.pkl"

    if os.path.exists(templates_file):
        print(f"Templates already exist at: {templates_file}")
        return True

    print("\nTemplates not found. Creating templates from handheld images...")
    print(f"Handheld directory: {handheld_dir}")

    image_files = sorted(handheld_dir.glob("*.jpg"))
    if not image_files:
        print(f"ERROR: No images found in {handheld_dir}")
        return False

    print(f"Found {len(image_files)} handheld images")

    # Create registrar
    registrar = ObjectRegistrar(
        detector=models['detector'],
        segmenter=models['segmenter'],
        extractor=models['extractor'],
        default_prompt="item held by hand",
        verbose=True
    )

    # Register all images
    instance_id = templates_path.stem  # Use directory name as instance ID
    print(f"\nRegistering instance: {instance_id}")

    results = registrar.register_batch(
        images=image_files,
        instance_id=instance_id,
        prompt="item held by hand"
    )

    # Check success
    num_success = sum(1 for r in results if r.success)
    if num_success == 0:
        print("ERROR: Failed to register any templates")
        return False

    # Save templates
    templates_path.parent.mkdir(parents=True, exist_ok=True)
    registrar.save_templates(templates_path)

    print(f"\nTemplates created successfully: {num_success}/{len(results)} images")
    return True


def filter_recognitions_by_best_match(recognitions: list, threshold: float = 0.0) -> list:
    """
    Filter recognitions to keep only the highest confidence match per instance ID.
    Only considers matches above the confidence threshold.

    Args:
        recognitions: List of recognition dictionaries
        threshold: Minimum confidence threshold

    Returns:
        Filtered list of recognitions (max 1 per instance ID)
    """
    best_matches = {}

    for rec in recognitions:
        # Skip low confidence detections
        if rec['confidence'] < threshold:
            continue

        inst_id = rec['instance_id']

        # Keep the one with higher confidence
        if inst_id not in best_matches or rec['confidence'] > best_matches[inst_id]['confidence']:
            best_matches[inst_id] = rec

    return list(best_matches.values())


def test_single_image_recognition(
    recognizer: ObjectRecognizer,
    image_path: Path,
    output_dir: Path,
    visualizer: RecognitionVisualizer,
    prompt: str = "objects"
) -> dict:
    """
    Test recognition on a single query image.

    Args:
        recognizer: ObjectRecognizer instance
        image_path: Path to query image
        output_dir: Output directory
        visualizer: RecognitionVisualizer instance
        prompt: Detection prompt

    Returns:
        Result dictionary
    """
    print("\n" + "=" * 80)
    print("TEST: SINGLE IMAGE RECOGNITION")
    print("=" * 80)
    print(f"Query image: {image_path.name}")
    print(f"Prompt: '{prompt}'")
    print("-" * 80)

    # Load image
    image = Image.open(image_path).convert('RGB')
    print(f"Image size: {image.size}")

    # Recognize objects
    result = recognizer.recognize(
        image=image,
        prompt=prompt,
        return_features=False,
        return_masks=True
    )

    # Print result
    print("-" * 80)
    print(f"Result: {result}")

    if result.success:
        # Filter results to find best match per instance
        best_matches = filter_recognitions_by_best_match(
            result.recognitions,
            threshold=recognizer.confidence_threshold
        )

        print(f"\nSUCCESS: Recognition complete!")
        print(f"  Raw Detections: {result.num_detections}")
        print(f"  Best Matches: {len(best_matches)}")

        # Print each recognition
        print(f"\nBest match details:")
        for i, rec in enumerate(best_matches):
            print(f"  [{i+1}] Instance: {rec['instance_id']}")
            print(f"      Confidence: {rec['confidence']:.4f}")
            print(f"      BBox: {[f'{x:.1f}' for x in rec['bbox']]}")
            print(f"      Seg score: {rec['seg_score']:.4f}")

            if 'top_k_matches' in rec:
                print(f"      Top-3 matches:")
                for rank, (match_id, score) in enumerate(rec['top_k_matches'][:3], 1):
                    print(f"        {rank}. {match_id}: {score:.4f}")
            print()

        # Save visualization (Best Matches Only)
        visualizer.save_recognition_visualization(
            image=image,
            recognitions=best_matches,
            image_id=f"single_{image_path.stem}",
            confidence_threshold=recognizer.confidence_threshold,
            show_top_k=False,
            show_masks=True
        )

        # Save visualization (with top-k, also filtered)
        visualizer.save_recognition_visualization(
            image=image,
            recognitions=best_matches,
            image_id=f"single_{image_path.stem}_topk",
            confidence_threshold=recognizer.confidence_threshold,
            show_top_k=True,
            show_masks=True
        )

        print(f"Visualizations saved to: {visualizer.viz_dir}")

    else:
        print(f"FAILED: {result.error_message}")

    print("=" * 80)

    # Build result dict for reporting
    result_dict = {
        'image_id': image_path.stem,
        'num_detections': result.num_detections,
        'recognitions': filter_recognitions_by_best_match(
            result.recognitions,
            threshold=recognizer.confidence_threshold
        ) if result.success else []
    }

    return result_dict


def test_batch_recognition(
    recognizer: ObjectRecognizer,
    image_paths: list,
    output_dir: Path,
    visualizer: RecognitionVisualizer,
    prompt: str = "objects"
) -> list:
    """
    Test recognition on multiple query images.

    Args:
        recognizer: ObjectRecognizer instance
        image_paths: List of paths to query images
        output_dir: Output directory
        visualizer: RecognitionVisualizer instance
        prompt: Detection prompt

    Returns:
        List of result dictionaries
    """
    print("\n" + "=" * 80)
    print(f"TEST: BATCH RECOGNITION ({len(image_paths)} images)")
    print("=" * 80)
    print(f"Prompt: '{prompt}'")
    print("-" * 80)

    results = []

    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] Processing {image_path.name}...")

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Recognize
        result = recognizer.recognize(
            image=image,
            prompt=prompt,
            return_features=False,
            return_masks=True
        )

        if result.success:
            # Filter results
            best_matches = filter_recognitions_by_best_match(
                result.recognitions,
                threshold=recognizer.confidence_threshold
            )

            print(f"  ✓ Raw Detections: {result.num_detections}")
            print(f"  ✓ Best Matches: {len(best_matches)}")

            # Save visualization
            visualizer.save_recognition_visualization(
                image=image,
                recognitions=best_matches,
                image_id=f"batch_{i+1:02d}_{image_path.stem}",
                confidence_threshold=recognizer.confidence_threshold,
                show_top_k=False,
                show_masks=True
            )

            recognitions_to_save = best_matches
        else:
            print(f"  ✗ Failed: {result.error_message}")
            recognitions_to_save = []

        # Build result dict
        result_dict = {
            'image_id': image_path.stem,
            'num_detections': result.num_detections,
            'recognitions': recognitions_to_save
        }
        results.append(result_dict)

    # Summary
    print("\n" + "-" * 80)
    print("BATCH RECOGNITION SUMMARY")
    print("-" * 80)
    total_detections = sum(r['num_detections'] for r in results)
    total_recognitions = sum(len(r['recognitions']) for r in results)

    print(f"Total images: {len(results)}")
    print(f"Total raw detections: {total_detections}")
    print(f"Total matched instances: {total_recognitions}")
    print(f"Avg matched per image: {total_recognitions/len(results):.2f}")
    print("=" * 80)

    return results


def main():
    """Main test function."""
    print("\n" + "=" * 80)
    print("OBJECT RECOGNIZER TEST SUITE")
    print("=" * 80)

    # Setup output directory
    output_dir = setup_output_directory("recognition_output")

    # Setup paths
    handheld_dir = PROJECT_ROOT / "examples" / "handheld" / "cellphone1"
    query_dir = PROJECT_ROOT / "examples" / "scence"  # Note: typo in directory name
    templates_path = PROJECT_ROOT / "debug_output" / "templates" / "cellphone1"

    # Check if query directory exists
    if not query_dir.exists():
        print(f"ERROR: Query directory not found: {query_dir}")
        return 1

    # Get query images
    query_files = sorted(query_dir.glob("*.jpg"))
    if not query_files:
        print(f"ERROR: No query images found in {query_dir}")
        return 1

    print(f"\nFound {len(query_files)} query images:")
    for img in query_files:
        print(f"  - {img.name}")

    # Load models
    print("\n" + "=" * 80)
    print("LOADING MODELS")
    print("=" * 80)

    loader = ModelLoader(device="cuda", verbose=True)
    models = loader.load_all()

    # Register templates if needed
    print("\n" + "=" * 80)
    print("CHECKING TEMPLATES")
    print("=" * 80)

    if not register_templates_if_needed(models, templates_path, handheld_dir):
        print("ERROR: Failed to create/load templates")
        return 1

    # Create recognizer
    print("\n" + "=" * 80)
    print("CREATING OBJECT RECOGNIZER")
    print("=" * 80)

    recognizer = ObjectRecognizer(
        detector=models['detector'],
        segmenter=models['segmenter'],
        extractor=models['extractor'],
        matcher=models['matcher'],
        default_prompt="objects",
        confidence_threshold=0.5,
        verbose=True
    )

    # Load templates
    num_loaded = recognizer.load_templates(templates_path)
    print(f"Loaded templates for {num_loaded} instance(s)")

    # Print template info
    template_info = recognizer.get_template_info()
    print(f"\nTemplate info:")
    print(f"  Instances: {template_info['num_instances']}")
    print(f"  Total templates: {template_info['total_templates']}")
    print(f"  Instance IDs: {template_info['instance_ids']}")

    # Create visualizer
    visualizer = RecognitionVisualizer(output_dir=output_dir, verbose=True)

    # Test 1: Single image recognition
    single_result = test_single_image_recognition(
        recognizer=recognizer,
        image_path=query_files[0],
        output_dir=output_dir,
        visualizer=visualizer,
        prompt="objects"
    )

    # Test 2: Batch recognition
    batch_results = test_batch_recognition(
        recognizer=recognizer,
        image_paths=query_files,
        output_dir=output_dir,
        visualizer=visualizer,
        prompt="objects"
    )

    # Generate reports
    print("\n" + "=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)

    all_results = [single_result] + batch_results

    visualizer.save_recognition_report(
        results=all_results,
        report_name="recognition_report"
    )

    visualizer.save_recognition_summary_json(
        results=all_results,
        summary_name="recognition_summary"
    )

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    total_images = len(all_results)
    total_detections = sum(r['num_detections'] for r in all_results)
    total_recognitions = sum(len(r['recognitions']) for r in all_results)

    print(f"Total images processed: {total_images}")
    print(f"Total detections: {total_detections}")
    print(f"Total recognitions: {total_recognitions}")
    print(f"Average detections per image: {total_detections/total_images:.2f}")
    print(f"\nOutput files saved to:")
    print(f"  {output_dir}")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
