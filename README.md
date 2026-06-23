## Installation

```bash
cd models
git clone https://github.com/IDEA-Research/Rex-Omni.git
cd Rex-Omni
conda create -n orr python=3.10 -y
conda activate orr
pip install torch==2.7.0 torchvision
pip install -r requirements.txt
pip install -v -e .
```