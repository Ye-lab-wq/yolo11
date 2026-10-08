# DetectDataset clean-v2 motion-blur benchmark

Deterministic derivative of the 625-image public test split. It is a secondary synthetic-corruption benchmark, not an independently collected dataset. Bounding boxes are unchanged. Light, moderate, and strong motion blur correspond to line-kernel lengths 3, 7, and 11 at a 640-pixel model input; native kernel sizes are resolution-scaled. Each source image receives one deterministic angle from 0/45/90/135 degrees, held fixed across severity. See `manifest.csv` for provenance and hashes.
