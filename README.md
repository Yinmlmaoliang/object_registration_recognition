# Object Registration and Recognition System

A few-shot object registration and recognition system that learns objects from handheld interaction videos and recognizes them in query scenes. The system uses four foundation models in a pipeline:

1. **RexOmni**: Object detection via text prompts
2. **SAM2**: Instance segmentation from bounding boxes
3. **DINOv3-MGFA**: Feature extraction with Mask-Guided Feature Aggregation
4. **SentenceTransformer**: Text attribute encoding for text-based retrieval

## Installation

All dependencies must be installed into the `orr` conda environment.

### 1. Create Conda Environment

```bash
conda create -n orr python=3.10 -y
conda activate orr
```

### 2. Install pip Dependencies

```bash
pip install -r requirements.txt
```

### 3. Clone and Install Rex-Omni

Clone Rex-Omni from [IDEA-Research/Rex-Omni](https://github.com/IDEA-Research/Rex-Omni) into the `models/` directory:

```bash
cd models
git clone https://github.com/IDEA-Research/Rex-Omni.git
cd Rex-Omni
pip install -e .
cd ../..
```

### 4. Clone and Install SAM2

Clone SAM2 from [facebookresearch/sam2](https://github.com/facebookresearch/sam2) into the `models/` directory:

```bash
cd models
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ../..
```

### 5. Download SAM2 Model Weights

SAM2 checkpoints must be downloaded into the `models/sam2/checkpoints/` directory:

```bash
cd models/sam2/checkpoints
bash download_ckpts.sh
cd ../../..
```

By default the system uses `sam2.1_hiera_large.pt`.

**Other model weights:**
- **Rex-Omni weights**: Should be in `models/Rex-Omni/checkpoints/` after installation
- **DINOv3 weights**: Downloaded automatically on first run via `torch.hub.load()` to `~/.cache/torch/hub/`
- **SentenceTransformer weights**: Downloaded automatically on first run from HuggingFace

### 6. Verify Installation

```bash
python tests/test_model_loader.py
```

## Quick Start

### Register Objects

```bash
# Register a handheld object to the template library
python register_object.py --image_dir examples/handheld/cellphone1/ \
    --instance_id cellphone1 --output templates/my_objects

# Add more instances to the same library (incremental)
python register_object.py --image_dir examples/handheld/mug1/ \
    --instance_id mug1 --output templates/my_objects

# With text attributes (enables text-based retrieval)
python register_object.py --image_dir examples/handheld/mug1/ \
    --instance_id mug1 --output templates/my_objects \
    --database examples/object_database.json

# With visualizations
python register_object.py --image_dir examples/handheld/cellphone1/ \
    --instance_id cellphone1 --output templates/my_objects --save_viz
```

### Recognize Objects

```bash
# Recognize objects from template library in query scenes
python recognize_object.py --image_dir examples/scence/ \
    --templates templates/my_objects

# Text-based object retrieval and recognition
python recognize_object.py --image_dir examples/scence/ \
    --templates templates/my_objects \
    --query "Find my Mickey Mouse mug"

# With custom thresholds
python recognize_object.py --image_dir examples/scence/ \
    --templates templates/my_objects \
    --threshold 0.6 --text_threshold 0.4
```

### Complete Workflow

```bash
# 1. Register objects (one or multiple instances)
python register_object.py --image_dir examples/handheld/cellphone1/ \
    --instance_id cellphone1 --output templates/my_objects

python register_object.py --image_dir examples/handheld/mug1/ \
    --instance_id mug1 --output templates/my_objects \
    --database examples/object_database.json

# 2. Recognize objects in query scenes
python recognize_object.py --image_dir examples/scence/ \
    --templates templates/my_objects

# 3. Or use text-based retrieval
python recognize_object.py --image_dir examples/scence/ \
    --templates templates/my_objects \
    --query "Find my Mickey Mouse mug"
```

## Project Structure

```
object_registration_recognition/
├── src/
│   ├── model_loader.py          # Unified model loading interface (lazy loading)
│   ├── registration/
│   │   └── object_registrar.py  # Object registration logic
│   ├── recognition/
│   │   └── object_recognizer.py # Object recognition logic
│   └── utils/
│       ├── rexomni_detector.py   # RexOmni detector wrapper
│       ├── sam2_segmenter.py     # SAM2 segmenter wrapper
│       ├── mgfa_extractor.py     # DINOv3-MGFA feature extraction
│       ├── text_encoder.py       # Text encoder (SentenceTransformer)
│       ├── matcher.py            # Template matching
│       ├── template_manager.py   # Template storage and management
│       ├── bbox_utils.py         # Bounding box utilities
│       └── result_analyzer.py    # Visualization and analysis
├── models/
│   └── dinov3.py                 # DINOv3 ViT wrapper (loaded via torch.hub)
├── examples/
│   ├── handheld/                 # Handheld object images
│   ├── scence/                   # Query scene images
│   └── object_database.json      # Object text attribute database
├── tests/
├── register_object.py            # Registration CLI tool
├── recognize_object.py           # Recognition CLI tool
├── requirements.txt              # Pip dependency list
└── README.md
```
