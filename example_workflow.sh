#!/bin/bash
# Example workflow demonstrating object registration and recognition

echo "========================================"
echo "Example Workflow: Object Registration and Recognition"
echo "========================================"

# Activate conda environment
echo ""
echo "Step 0: Activate conda environment"
echo "----------------------------------------"
# conda activate rexomni  # Uncomment if needed

# Step 1: Register object from handheld images
echo ""
echo "Step 1: Register object instance to template library"
echo "----------------------------------------"
python register_object.py \
  --image_dir examples/handheld/cellphone1/ \
  --instance_id cellphone1 \
  --output templates/my_objects \
  --save_viz

# Check if registration succeeded
if [ $? -ne 0 ]; then
    echo "Registration failed! Exiting."
    exit 1
fi

echo ""
echo "Note: You can add more object instances to the same template library:"
echo "  python register_object.py --image_dir examples/handheld/mug/ --instance_id mug1 --output templates/my_objects"

# Step 2: Recognize object in query scenes
echo ""
echo "Step 2: Recognize objects from template library in query scenes"
echo "----------------------------------------"
python recognize_object.py \
  --image_dir examples/scence/ \
  --templates templates/my_objects \
  --output recognition_results/ \
  --threshold 0.5

# Check if recognition succeeded
if [ $? -ne 0 ]; then
    echo "Recognition failed! Exiting."
    exit 1
fi

echo ""
echo "========================================"
echo "Workflow completed successfully!"
echo "========================================"
echo ""
echo "Results:"
echo "  - Template library: templates/my_objects_templates.pkl (contains cellphone1)"
echo "  - Template metadata: templates/my_objects_metadata.json"
echo "  - Registration visualizations: templates/my_objects_visualizations/"
echo "  - Recognition visualizations: recognition_results/visualizations/"
echo ""
echo "To add more objects to the same template library, use:"
echo "  python register_object.py --image_dir <dir> --instance_id <new_id> --output templates/my_objects"
echo ""
