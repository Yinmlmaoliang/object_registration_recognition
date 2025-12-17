# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a few-shot object registration and recognition system that learns objects from handheld interaction videos and recognizes them in query scenes. The system uses three foundation models in a pipeline:

1. **RexOmni**: Object detection via text prompts
2. **SAM2**: Instance segmentation from bounding boxes
3. **DINOv3-FFA**: Feature extraction with Foreground Feature Averaging

The core workflow is:
- **Registration Phase**: Process handheld images → detect objects → segment → extract features → build template library
- **Recognition Phase** (TODO): Match query objects to registered templates using similarity metrics

## Development Environment

**Conda Environment**: `rexomni`
- The project expects to run in the `rexomni` conda environment
- Tests explicitly check for this environment

**Device**: CUDA (GPU) is expected for model inference
- Models are configured to run on CUDA by default
- All model wrappers support `device` parameter

## Key Commands

### Application Scripts

**Object Registration**
```bash
# Register objects to a shared template library (use unified library name)
python register_object.py --image_dir examples/handheld/cellphone1/ --instance_id cellphone1 --output templates/my_objects

# Add more instances to the same library (incremental)
python register_object.py --image_dir examples/handheld/mug/ --instance_id mug1 --output templates/my_objects

# With visualizations
python register_object.py --image_dir examples/handheld/cellphone1/ --instance_id cellphone1 --output templates/my_objects --save_viz
```

**Object Recognition**
```bash
# Recognize objects from template library in query scenes
python recognize_object.py --image_dir examples/scence/ --templates templates/my_objects

# With custom threshold
python recognize_object.py --image_dir examples/scence/ --templates templates/my_objects --threshold 0.6
```

**Complete Workflow**
```bash
# Run example workflow script
./example_workflow.sh
```

See [QUICKSTART.md](QUICKSTART.md) for quick start guide and [SCRIPTS_README.md](SCRIPTS_README.md) for detailed documentation.

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python tests/test_model_loader.py

# Run test with verbose output
python -m pytest tests/test_model_loader.py -v
```

### Model Setup
The system requires downloading SAM2 checkpoints:
```bash
cd models/sam2/checkpoints
bash download_ckpts.sh
```

## Architecture

### Application Scripts

**`register_object.py`**: Command-line tool for object registration
- Registers objects from handheld images via command-line interface
- Supports single image or batch processing
- **Key feature**: Uses unified template library name (`--output`), distinguishes instances by `--instance_id`
- Implements incremental template addition (can add new instances to existing template library)
- Optional visualization output
- Usage: `python register_object.py --image_dir <dir> --instance_id <id> --output templates/my_objects [--save_viz]`

**`recognize_object.py`**: Command-line tool for object recognition
- Recognizes registered objects in query scenes
- Loads templates and performs detection → segmentation → matching
- Outputs matches with confidence scores to terminal
- Saves visualization results with bounding boxes and masks
- Usage: `python recognize_object.py --image_dir <dir> --templates <path>`

**`example_workflow.sh`**: Example workflow script
- Demonstrates complete registration and recognition pipeline
- Can be used as a template for custom workflows

### Module Structure

**`src/model_loader.py`**: Central model loading interface
- **Lazy loading pattern**: Models are only loaded when accessed
- Use `ModelLoader.load_all()` to get all models at once
- Access individual models via properties: `.detector`, `.segmenter`, `.extractor`, `.matcher`
- Default paths for models are defined as class constants

**`src/registration/`**: Object registration logic
- `ObjectRegistrar`: Main class for registering objects from handheld images
- Pipeline: detect → segment → extract features → store templates
- Templates stored as numpy arrays with associated metadata
- Supports batch registration and save/load to disk

**`src/recognition/`**: Object recognition logic
- **Status**: Not yet implemented (placeholder module)
- **TODO**: Implement ObjectRecognizer class for query-time matching

**`src/utils/`**: Core model wrappers and utilities
- `rexomni_detector.py`: RexOmni detection wrapper with prompt strategies
- `sam2_segmenter.py`: SAM2 segmentation wrapper supporting single/batch modes
- `ffa_extractor.py`: DINOv3 feature extractor with Foreground Feature Averaging
- `matcher.py`: Template matching with cosine/euclidean similarity
- `template_manager.py`: Template storage and management
- `result_analyzer.py`: Visualization and analysis tools for detection/segmentation results
- `bbox_utils.py`: Bounding box utilities (IoU, conversion)

### Model Loading Pattern

All models use lazy loading to avoid loading unnecessary models:

```python
loader = ModelLoader(device="cuda")
models = loader.load_all()  # Returns dict: {detector, segmenter, extractor, matcher}

# Or load individually
detector = loader.detector      # Auto-loads on first access
segmenter = loader.segmenter
```

### Feature Extraction with FFA

**Critical Detail**: When using `FFAFeatureExtractor.extract_embedding()`:
- Masks must be in **FULL IMAGE SIZE** (H_full, W_full)
- The extractor automatically crops masks to match the bbox region
- This ensures proper spatial alignment with DINOv3 patch features
- The cropped region is resized to `image_size` (default 448x448)
- Features are extracted at patch-level then aggregated via FFA

### Detection Pipeline Details

**RexOmni Detector**:
- Uses pip-installed `rex_omni` package (`RexOmniWrapper`)
- Default prompt: "item held by hand"
- Multiple prompt strategies available in `PROMPT_STRATEGIES` dict
- Returns bounding boxes in [x0, y0, x1, y1] format

**SAM2 Segmenter**:
- Built from local `models/sam2/` submodule
- Uses SAM2ImagePredictor for bbox-to-mask conversion
- Supports single, multiple, and batch segmentation modes
- Returns binary masks (H, W) and confidence scores

**Template Matching**:
- Cosine similarity (default) or Euclidean distance
- Aggregation strategies: 'max', 'mean', 'avg_k' for multi-template instances
- Returns top-k matches with scores per proposal

### Result Analysis & Visualization

**`ResultAnalyzer` Features**:
- **Directory Structure**: Automatically organizes outputs into `visualizations`, `metrics`, `logs`, and `error_analysis`
- **Visualization**:
  - Draws bounding boxes (GT=Green, Pred=Red/Orange/Lime based on IoU)
  - Overlays segmentation masks with transparency and contours
  - Handles multi-object scenarios with matching lines between Pred and GT
  - Visualizes False Positives (FP) and False Negatives (FN)
- **Reporting**:
  - Generates detailed logs per prompt strategy
  - Creates CSV/Markdown summary tables for metrics
  - JSON exports for programmatic analysis
  - specialized error analysis for low IoU cases (< 0.3)

## File Paths

Model paths (relative to project root):
- RexOmni: `models/Rex-Omni/`
- SAM2 checkpoint: `models/sam2/checkpoints/sam2.1_hiera_large.pt`
- SAM2 config: `models/sam2/configs/sam2.1/sam2.1_hiera_l.yaml`
- DINOv3: loaded via `models/dinov3.py` (uses `Dinov3ViT` class)

Example data:
- Handheld images: `examples/handheld/cellphone1/`
- Query scenes: `examples/scence/` (note: typo in directory name)

## Important Implementation Details

### Path Management
- `PROJECT_ROOT` is computed as parent of project root in model wrappers
- SAM2 path is added to `sys.path` dynamically in `sam2_segmenter.py`
- DINOv3 imports from `models/dinov3.py`

### Registration Workflow
The `ObjectRegistrar` encapsulates the full pipeline:

1. Load image (PIL or path)
2. Detect with RexOmni (returns bboxes)
3. If multiple detections and ground truth provided, select best by IoU
4. Segment with SAM2 using selected bbox
5. Extract embedding with DINOv3-FFA using mask
6. Store embedding in templates dict keyed by instance_id

Templates can be saved as pickle (embeddings) + JSON (metadata).

### Testing Strategy
Tests use mocking extensively to avoid loading actual models:
- `@patch` decorators mock model classes from utils modules
- Tests verify lazy loading behavior and caching
- Tests check that correct parameters are passed to model constructors

## Known Issues

- Recognition module is not implemented yet
- Scene example directory has typo: `scence` instead of `scene`
- No requirements.txt or environment.yml file in project root

## Model Dependencies

External models in `models/`:
- `sam2/`: SAM2 repository (with full training/demo code)
- `Rex-Omni/`: RexOmni model weights
- `dinov3.py`: DINOv3 wrapper (class `Dinov3ViT`)
