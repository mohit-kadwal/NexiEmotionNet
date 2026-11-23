# Data

This repository does **not** contain any dataset. RAF-DB is distributed by its authors on request under its own
license and must not be redistributed.

## RAF-DB (used in all reported experiments)

Li, S., Deng, W., and Du, J., "Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression
Recognition in the Wild," CVPR 2017.

1. Request the dataset from its authors (search for the "Real-world Affective Faces Database" project page and
   follow the request instructions).
2. Place the archive `RAF DB Dataset.zip` in the same folder as `NexiEmotionNet_experiments.ipynb`.
   The notebook extracts it once into `dataset/`.
3. After extraction the folder layout must be:

```
dataset/
└── DATASET/
    ├── train/
    │   ├── 1/   (Surprise,  1,290 images)
    │   ├── 2/   (Fear,        281 images)
    │   ├── 3/   (Disgust,     717 images)
    │   ├── 4/   (Happiness, 4,772 images)
    │   ├── 5/   (Sadness,   1,982 images)
    │   ├── 6/   (Anger,       705 images)
    │   └── 7/   (Neutral,   2,524 images)
    └── test/
        ├── 1/ ... 7/   (329, 74, 160, 1,185, 478, 162 and 680 images)
```

The split is used as provided: 12,271 training images and 3,068 test images (15,339 in total). The folder
names 1–7 follow the RAF-DB label order and are mapped to the class names in the notebook
(`CLASS_NAMES = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]`).
