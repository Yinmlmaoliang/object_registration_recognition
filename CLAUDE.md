# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a few-shot object registration and recognition system that learns objects from handheld interaction videos and recognizes them in query scenes. The system uses four foundation models in a pipeline:

1. **RexOmni**: Object detection via text prompts
2. **SAM2**: Instance segmentation from bounding boxes
3. **DINOv3-FFA**: Feature extraction with Foreground Feature Averaging
4. **SentenceTransformer**: Text attribute encoding for RAG-based retrieval

The core workflow is:
- **Registration Phase**: Process handheld images → detect objects → segment → extract visual features → encode text attributes → build template library
- **Recognition Phase**: Text-based retrieval → visual detection → segmentation → feature matching → object localization

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

# With text attributes from database (enables text-based retrieval)
python register_object.py --image_dir examples/handheld/mug1/ --instance_id mug1 --output templates/my_objects --database examples/object_database.json
```

**Object Recognition**
```bash
# Recognize objects from template library in query scenes
python recognize_object.py --image_dir examples/scence/ --templates templates/my_objects

# With custom threshold
python recognize_object.py --image_dir examples/scence/ --templates templates/my_objects --threshold 0.6

# Text-based object retrieval and recognition
python recognize_object.py --image_dir examples/scence/ --templates templates/my_objects --query "Find my Mickey Mouse mug"

# With custom text similarity threshold
python recognize_object.py --image_dir examples/scence/ --templates templates/my_objects --query "my favorite mug" --text_threshold 0.4
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
- **Text attributes**: Use `--database` to load attributes from JSON for text-based retrieval
- Optional visualization output
- Usage: `python register_object.py --image_dir <dir> --instance_id <id> --output templates/my_objects [--database <json>] [--save_viz]`

**`recognize_object.py`**: Command-line tool for object recognition
- Recognizes registered objects in query scenes
- Loads templates and performs detection → segmentation → matching
- **Text query**: Use `--query` for natural language object retrieval
- Outputs matches with confidence scores to terminal
- Saves visualization results with bounding boxes and masks
- Usage: `python recognize_object.py --image_dir <dir> --templates <path> [--query "text query"]`

**`example_workflow.sh`**: Example workflow script
- Demonstrates complete registration and recognition pipeline
- Can be used as a template for custom workflows

### Module Structure

**`src/model_loader.py`**: Central model loading interface
- **Lazy loading pattern**: Models are only loaded when accessed
- Use `ModelLoader.load_all()` to get all models at once
- Use `ModelLoader.load_all(include_text_encoder=True)` to include text encoder
- Access individual models via properties: `.detector`, `.segmenter`, `.extractor`, `.matcher`, `.text_encoder`
- Default paths for models are defined as class constants

**`src/registration/`**: Object registration logic
- `ObjectRegistrar`: Main class for registering objects from handheld images
- Pipeline: detect → segment → extract visual features → encode text attributes → store templates
- Templates stored as numpy arrays with associated metadata (unified format)
- Supports batch registration, save/load to disk, and text attribute encoding
- Key methods: `register()`, `register_with_attributes()`, `save_templates()`, `load_templates()`

**`src/recognition/`**: Object recognition logic
- `ObjectRecognizer`: Main class for recognizing objects in query scenes
- Pipeline: detect → segment → extract features → match against templates
- Supports text-based retrieval via `retrieve_by_text()` and `recognize_with_text_query()`
- Key methods: `recognize()`, `retrieve_by_text()`, `recognize_with_text_query()`

**`src/utils/`**: Core model wrappers and utilities
- `rexomni_detector.py`: RexOmni detection wrapper with prompt strategies
- `sam2_segmenter.py`: SAM2 segmentation wrapper supporting single/batch modes
- `ffa_extractor.py`: DINOv3 feature extractor with Foreground Feature Averaging
- `text_encoder.py`: SentenceTransformer wrapper for text attribute encoding
- `matcher.py`: Template matching with cosine/euclidean similarity
- `template_manager.py`: Template storage and management
- `result_analyzer.py`: Visualization and analysis tools for detection/segmentation results
- `bbox_utils.py`: Bounding box utilities (IoU, conversion)

### Model Loading Pattern

All models use lazy loading to avoid loading unnecessary models:

```python
loader = ModelLoader(device="cuda")
models = loader.load_all()  # Returns dict: {detector, segmenter, extractor, matcher}

# Include text encoder for text-based retrieval
models = loader.load_all(include_text_encoder=True)  # Also includes text_encoder

# Or load individually
detector = loader.detector      # Auto-loads on first access
segmenter = loader.segmenter
text_encoder = loader.text_encoder  # Loads SentenceTransformer
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

**Text Encoder**:
- Uses SentenceTransformer `all-MiniLM-L6-v2` model
- 384-dimensional embeddings
- Max pooling aggregation for multiple attribute descriptions
- Cosine similarity for text-based retrieval

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
- TextEncoder: loaded via HuggingFace (`all-MiniLM-L6-v2`)

Example data:
- Handheld images: `examples/handheld/cellphone1/`
- Query scenes: `examples/scence/` (note: typo in directory name)
- Object database: `examples/object_database.json` (text attributes)

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
5. Extract visual embedding with DINOv3-FFA using mask
6. (Optional) Encode text attributes with SentenceTransformer
7. Store embeddings in templates dict keyed by instance_id

Templates saved in unified format:
```python
# templates.pkl
{
    'visual_embeddings': {instance_id: np.ndarray[N, 768]},  # DINOv3 features
    'text_embeddings': {instance_id: np.ndarray[384]}        # Text embeddings
}

# metadata.json
{
    instance_id: {
        'visual_metadata': [{bbox, seg_score, prompt}, ...],
        'attributes': ["attribute 1", "attribute 2", ...]
    }
}
```

### Text-Based Retrieval Workflow
The `ObjectRecognizer` supports text-based retrieval:

1. Encode query text with SentenceTransformer
2. Compute cosine similarity with all text embeddings
3. Filter candidates above threshold
4. Perform visual recognition on filtered candidates
5. Return matched objects with both visual and text scores

```python
# Example usage
recognizer.retrieve_by_text(query="my favorite mug", text_encoder=encoder, top_k=5)
recognizer.recognize_with_text_query(image, query="my favorite mug", text_encoder=encoder)
```

### Testing Strategy
Tests use mocking extensively to avoid loading actual models:
- `@patch` decorators mock model classes from utils modules
- Tests verify lazy loading behavior and caching
- Tests check that correct parameters are passed to model constructors

## Known Issues

- Scene example directory has typo: `scence` instead of `scene`
- No requirements.txt or environment.yml file in project root

## Dependencies

Additional Python packages required:
- `sentence-transformers`: For text attribute encoding (`pip install sentence-transformers`)

## Model Dependencies

External models in `models/`:
- `sam2/`: SAM2 repository (with full training/demo code)
- `Rex-Omni/`: RexOmni model weights
- `dinov3.py`: DINOv3 wrapper (class `Dinov3ViT`)
