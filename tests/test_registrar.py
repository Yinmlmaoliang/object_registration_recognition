#!/usr/bin/env python3
"""
Test script for ObjectRegistrar functionality.

This script tests the object registration pipeline including:
- One-shot registration (single image)
- N-shot registration (multiple images)
- Visualization of detection and segmentation results
- Template saving and loading
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
from src.utils.result_analyzer import (
    draw_detection_with_segmentation,
    ResultAnalyzer
)


def setup_output_directory(base_dir: str = "debug_output") -> Path:
    """
    Setup output directory structure for debugging.

    Args:
        base_dir: Base directory name

    Returns:
        Path to output directory
    """
    output_dir = PROJECT_ROOT / base_dir
    output_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (output_dir / "one_shot").mkdir(exist_ok=True)
    (output_dir / "n_shot").mkdir(exist_ok=True)
    (output_dir / "templates").mkdir(exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"  - one_shot/: Single image registration visualizations")
    print(f"  - n_shot/: Multiple image registration visualizations")
    print(f"  - templates/: Saved template files\n")

    return output_dir


def save_registration_debug(
    image: Image.Image,
    result: dict,
    output_path: Path,
    image_name: str
):
    """
    Save debug visualization for a registration result.

    Args:
        image: Original PIL Image
        result: Dictionary with bbox, mask, and other info
        output_path: Path to save visualization
        image_name: Name for the output file
    """
    # Extract info from result
    bbox = result.get('bbox')
    mask = result.get('mask')
    seg_score = result.get('seg_score', 0.0)

    if bbox is None:
        print(f"  Warning: No bbox for {image_name}, skipping visualization")
        return

    # For registration, we use the detected bbox as both pred and GT
    # This shows what was actually detected and segmented
    img_with_result = draw_detection_with_segmentation(
        image=image,
        pred_bbox=bbox,
        gt_bbox=bbox,  # Use same bbox as GT for visualization
        mask=mask,
        iou=1.0,  # Perfect match since pred==GT
        prompt_text=f"Registered (seg_score={seg_score:.3f})",
        mask_color=(0, 255, 0),
        mask_alpha=0.4,
        show_labels=True
    )

    # Save visualization
    save_path = output_path / f"{image_name}.jpg"
    img_with_result.save(save_path, quality=95)
    print(f"  Saved visualization: {save_path}")


def test_one_shot_registration(
    registrar: ObjectRegistrar,
    image_path: Path,
    instance_id: str,
    output_dir: Path,
    prompt: str = "item held by hand"
) -> bool:
    """
    Test one-shot registration (single image).

    Args:
        registrar: ObjectRegistrar instance
        image_path: Path to the handheld image
        instance_id: Instance identifier
        output_dir: Output directory for visualizations
        prompt: Detection prompt

    Returns:
        True if registration succeeded
    """
    print("\n" + "=" * 80)
    print("TEST 1: ONE-SHOT REGISTRATION")
    print("=" * 80)
    print(f"Image: {image_path.name}")
    print(f"Instance ID: {instance_id}")
    print(f"Prompt: '{prompt}'")
    print("-" * 80)

    # Load image
    image = Image.open(image_path).convert('RGB')

    # Register the object
    result = registrar.register(
        image=image,
        instance_id=instance_id,
        prompt=prompt
    )

    # Print result
    print("-" * 80)
    print(f"Result: {result}")

    if result.success:
        print(f"SUCCESS: Object '{instance_id}' registered successfully!")
        print(f"  Feature dimension: {result.feature_dim}")
        print(f"  Detection bbox: {result.detection_bbox}")
        print(f"  Segmentation score: {result.segmentation_score:.4f}")

        # Save debug visualization
        # Get the mask and bbox from the last registration
        metadata = registrar._metadata[instance_id][-1]

        # Perform detection and segmentation again for visualization
        # (in production, you'd save these during registration)
        det_result = registrar.detector.detect(image, prompt)
        if det_result['success']:
            bbox = det_result['predictions'][0]
            seg_result = registrar.segmenter.segment_single(image, bbox)

            if seg_result['success']:
                debug_info = {
                    'bbox': bbox,
                    'mask': seg_result['mask'],
                    'seg_score': seg_result['score']
                }

                save_registration_debug(
                    image=image,
                    result=debug_info,
                    output_path=output_dir / "one_shot",
                    image_name=f"{instance_id}_{image_path.stem}"
                )
    else:
        print(f"FAILED: {result.error_message}")

    print("=" * 80)
    return result.success


def test_n_shot_registration(
    registrar: ObjectRegistrar,
    image_paths: list,
    instance_id: str,
    output_dir: Path,
    prompt: str = "item held by hand"
) -> tuple:
    """
    Test n-shot registration (multiple images).

    Args:
        registrar: ObjectRegistrar instance
        image_paths: List of paths to handheld images
        instance_id: Instance identifier
        output_dir: Output directory for visualizations
        prompt: Detection prompt

    Returns:
        Tuple of (num_success, num_total)
    """
    print("\n" + "=" * 80)
    print(f"TEST 2: N-SHOT REGISTRATION (N={len(image_paths)})")
    print("=" * 80)
    print(f"Instance ID: {instance_id}")
    print(f"Number of images: {len(image_paths)}")
    print(f"Prompt: '{prompt}'")
    print("-" * 80)

    # Register all images
    results = []
    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] Processing {image_path.name}...")

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Register
        result = registrar.register(
            image=image,
            instance_id=instance_id,
            prompt=prompt
        )

        results.append(result)

        # Save debug visualization if successful
        if result.success:
            # Re-run detection and segmentation for visualization
            det_result = registrar.detector.detect(image, prompt)
            if det_result['success']:
                bbox = det_result['predictions'][0]
                seg_result = registrar.segmenter.segment_single(image, bbox)

                if seg_result['success']:
                    debug_info = {
                        'bbox': bbox,
                        'mask': seg_result['mask'],
                        'seg_score': seg_result['score']
                    }

                    save_registration_debug(
                        image=image,
                        result=debug_info,
                        output_path=output_dir / "n_shot",
                        image_name=f"{instance_id}_img{i+1:02d}_{image_path.stem}"
                    )

    # Summary
    num_success = sum(1 for r in results if r.success)
    num_total = len(results)

    print("\n" + "-" * 80)
    print("N-SHOT REGISTRATION SUMMARY")
    print("-" * 80)
    print(f"Total images: {num_total}")
    print(f"Successful: {num_success}")
    print(f"Failed: {num_total - num_success}")
    print(f"Success rate: {num_success/num_total*100:.1f}%")

    # Show per-image results
    print("\nPer-image results:")
    for i, (path, result) in enumerate(zip(image_paths, results)):
        status = "✓" if result.success else "✗"
        print(f"  {status} {path.name}: {result}")

    print("=" * 80)

    return num_success, num_total


def test_template_save_load(
    registrar: ObjectRegistrar,
    output_dir: Path
) -> bool:
    """
    Test template saving and loading.

    Args:
        registrar: ObjectRegistrar instance with registered templates
        output_dir: Output directory for template files

    Returns:
        True if save/load succeeded
    """
    print("\n" + "=" * 80)
    print("TEST 3: TEMPLATE SAVE/LOAD")
    print("=" * 80)

    # Get template statistics before saving
    stats_before = registrar.get_template_statistics()
    print("\nTemplate statistics before save:")
    print(f"  Instances: {stats_before['num_instances']}")
    print(f"  Total templates: {stats_before['total_templates']}")
    print(f"  Feature dimension: {stats_before['feature_dim']}")
    print(f"  Templates per instance:")
    for inst_id, count in stats_before['templates_per_instance'].items():
        print(f"    - {inst_id}: {count} templates")

    # Save templates
    save_path = output_dir / "templates" / "cellphone1"
    print(f"\nSaving templates to: {save_path}")

    try:
        templates_path, metadata_path = registrar.save_templates(save_path)
        print(f"  ✓ Templates saved: {templates_path}")
        print(f"  ✓ Metadata saved: {metadata_path}")
    except Exception as e:
        print(f"  ✗ Save failed: {e}")
        return False

    # Create a new registrar and load templates
    print("\nTesting template loading...")
    from src.model_loader import ModelLoader
    loader = ModelLoader(device="cuda", verbose=False)
    models = loader.load_all()

    new_registrar = ObjectRegistrar(
        detector=models['detector'],
        segmenter=models['segmenter'],
        extractor=models['extractor'],
        verbose=False
    )

    try:
        num_loaded = new_registrar.load_templates(save_path)
        print(f"  ✓ Loaded {num_loaded} instance(s)")
    except Exception as e:
        print(f"  ✗ Load failed: {e}")
        return False

    # Get template statistics after loading
    stats_after = new_registrar.get_template_statistics()
    print("\nTemplate statistics after load:")
    print(f"  Instances: {stats_after['num_instances']}")
    print(f"  Total templates: {stats_after['total_templates']}")
    print(f"  Feature dimension: {stats_after['feature_dim']}")

    # Verify they match
    match = (
        stats_before['num_instances'] == stats_after['num_instances'] and
        stats_before['total_templates'] == stats_after['total_templates']
    )

    if match:
        print("\n✓ Save/Load verification: PASSED")
    else:
        print("\n✗ Save/Load verification: FAILED (statistics don't match)")

    print("=" * 80)
    return match


def main():
    """Main test function."""
    print("\n" + "=" * 80)
    print("OBJECT REGISTRAR TEST SUITE")
    print("=" * 80)

    # Setup output directory
    output_dir = setup_output_directory("debug_output")

    # Setup paths
    handheld_dir = PROJECT_ROOT / "examples" / "handheld" / "cellphone1"
    image_files = sorted(handheld_dir.glob("*.jpg"))

    if not image_files:
        print(f"ERROR: No images found in {handheld_dir}")
        return 1

    print(f"Found {len(image_files)} handheld images:")
    for img in image_files:
        print(f"  - {img.name}")

    # Load models
    print("\n" + "=" * 80)
    print("LOADING MODELS")
    print("=" * 80)

    loader = ModelLoader(device="cuda", verbose=True)
    models = loader.load_all()

    # Create registrar
    print("\n" + "=" * 80)
    print("CREATING OBJECT REGISTRAR")
    print("=" * 80)

    registrar = ObjectRegistrar(
        detector=models['detector'],
        segmenter=models['segmenter'],
        extractor=models['extractor'],
        default_prompt="item held by hand",
        verbose=True
    )

    # Test 1: One-shot registration (first image)
    one_shot_success = test_one_shot_registration(
        registrar=registrar,
        image_path=image_files[0],
        instance_id="cellphone1",
        output_dir=output_dir,
        prompt="item held by hand"
    )

    # Clear templates before n-shot test
    registrar.clear_templates()

    # Test 2: N-shot registration (all images)
    n_shot_success, n_shot_total = test_n_shot_registration(
        registrar=registrar,
        image_paths=image_files,
        instance_id="cellphone1",
        output_dir=output_dir,
        prompt="item held by hand"
    )

    # Test 3: Template save/load
    save_load_success = test_template_save_load(
        registrar=registrar,
        output_dir=output_dir
    )

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Test 1 - One-shot registration: {'PASSED ✓' if one_shot_success else 'FAILED ✗'}")
    print(f"Test 2 - N-shot registration: {n_shot_success}/{n_shot_total} successful")
    print(f"Test 3 - Template save/load: {'PASSED ✓' if save_load_success else 'FAILED ✗'}")
    print("\nOutput files saved to:")
    print(f"  {output_dir}")
    print("=" * 80 + "\n")

    # Return 0 if all tests passed
    all_passed = (
        one_shot_success and
        n_shot_success == n_shot_total and
        save_load_success
    )

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
