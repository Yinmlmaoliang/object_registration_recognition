import json
import os
import sys
import numpy as np
from pathlib import Path

def inspect_templates(template_dir="templates", template_name="objects"):
    """
    Inspect and visualize the contents of saved templates.
    """
    base_path = Path(template_dir) / template_name
    metadata_path = base_path.with_name(f"{template_name}_metadata.json")
    embeddings_path = base_path.with_name(f"{template_name}_templates.pkl")
    
    print(f"Inspecting templates at: {base_path.parent}")
    print(f"{'-'*50}")

    # 1. Load Metadata
    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Found metadata file: {metadata_path.name}")
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return

    # 2. Load Embeddings (if available)
    embeddings = None
    if embeddings_path.exists():
        try:
            import pickle
            with open(embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
            print(f"Found embeddings file: {embeddings_path.name}")
        except Exception as e:
            print(f"Error reading embeddings: {e}")
    else:
        print(f"Warning: Embeddings file not found at {embeddings_path}")
        print("Cannot visualize vector distributions.")

    print(f"{'-'*50}")
    print(f"Registered Objects Summary:")
    print(f"{'-'*50}")

    all_visual_features = []
    all_labels = []
    
    for instance_id, data in metadata.items():
        print(f"Object ID: {instance_id}")
        
        # Visual samples info
        if isinstance(data, dict) and 'visual_metadata' in data:
            num_samples = len(data['visual_metadata'])
            attributes = data.get('attributes', [])
            
            print(f"  - Visual Templates: {num_samples} samples")
            
            # Show attributes
            if attributes:
                print(f"  - Text Attributes:")
                for attr in attributes:
                    print(f"    * \"{attr}\"")
            else:
                print(f"  - Text Attributes: None")
            
            # Show vector info if available
            if embeddings:
                # Check for visual embeddings
                if 'visual_embeddings' in embeddings and instance_id in embeddings['visual_embeddings']:
                    vis_emb = embeddings['visual_embeddings'][instance_id]
                    # Handle if it's a list or numpy array
                    if isinstance(vis_emb, list):
                        vis_emb = np.stack(vis_emb)
                    print(f"  - Visual Embeddings Shape: {vis_emb.shape}")
                    
                    # Collect for visualization
                    for i in range(vis_emb.shape[0]):
                        all_visual_features.append(vis_emb[i])
                        all_labels.append(instance_id)

                # Check for text embeddings
                if 'text_embeddings' in embeddings and instance_id in embeddings['text_embeddings']:
                    text_emb = embeddings['text_embeddings'][instance_id]
                    print(f"  - Text Embedding Shape: {text_emb.shape}")
        
        print("")

    print(f"{'-'*50}")
    
    # 3. Visualization
    if embeddings and all_visual_features:
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            
            print("Generating visualization...")
            X = np.array(all_visual_features)
            
            # We need at least 2 samples for PCA
            if X.shape[0] < 2:
                print("Not enough samples for PCA visualization.")
                return

            n_components = min(2, X.shape[0])
            pca = PCA(n_components=n_components)
            X_r = pca.fit_transform(X)
            
            plt.figure(figsize=(10, 8))
            
            # Get unique labels
            unique_labels = list(set(all_labels))
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            
            for color, label in zip(colors, unique_labels):
                indices = [i for i, x in enumerate(all_labels) if x == label]
                # If 1D PCA (only 2 samples), plot on x-axis with y=0
                if n_components == 1:
                    plt.scatter(X_r[indices, 0], np.zeros_like(indices), color=color, alpha=0.8, lw=2, label=label)
                else:
                    plt.scatter(X_r[indices, 0], X_r[indices, 1], color=color, alpha=0.8, lw=2, label=label)
            
            plt.title('PCA visualization of Object Visual Embeddings (DINOv3)')
            plt.legend(loc='best', shadow=False, scatterpoints=1)
            plt.grid(True, alpha=0.3)
            
            output_file = 'templates_visualization.png'
            plt.savefig(output_file)
            print(f"Visualization saved to: {output_file}")
            
        except ImportError:
            print("matplotlib or sklearn not installed. Skipping visualization.")
        except Exception as e:
            print(f"Error generating visualization: {e}")

if __name__ == "__main__":
    template_name = "objects"
    if len(sys.argv) > 1:
        template_name = sys.argv[1]
    
    inspect_templates(template_name=template_name)
