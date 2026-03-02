import os
from rfdetr import RFDETRBase

DATASET_DIR = os.environ.get("DATASET_DIR", "test.v1-test.coco")

# Start with base pretrained weights
model = RFDETRBase()

model.train(
    dataset_dir=DATASET_DIR,
    epochs=25,
    batch_size=4,      # V100 16GB safe starting point
    lr=1e-4,
)

model.export("runs/rfdetr_waterpolo")