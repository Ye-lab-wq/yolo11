# DUT Anti-UAV paired reproduction

This directory contains only the reproducibility material added for the DUT Anti-UAV experiment.
The official ZIP files and extracted images remain outside Git under
`/media/b520/KESU/Dataset/DUT-Anti-UAV`.

Protocol:

- official release split: 5,200 train / 2,600 val / 2,200 test images;
- one class: `UAV`;
- YOLO11n and `yolo11n-MEDA-Pro.yaml` use the same seed, image size, batch size,
  optimizer, maximum epochs and early-stopping patience;
- validation selects `best.pt`; the test split is evaluated once after model selection;
- both models start from random initialization because there is no architecture-matched
  public pretrained MEDA-Pro checkpoint for DUT.

Data preparation:

```bash
/home/b520/anaconda3/envs/YHP/bin/python reproduction/dut/prepare_dut.py \
  --source /media/b520/KESU/Dataset/DUT-Anti-UAV/original \
  --output /media/b520/KESU/Dataset/DUT-Anti-UAV/yolo
```
