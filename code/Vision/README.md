# Vision source code

All files here are our actual source code; this directory can be copied as a
whole.

## Contents

- `vision/yolo/train.py`: YOLO11 training with the settings we used (100
  epochs, 960 input size, batch 8, Adam, patience 20).
- `vision/yolo/infer_base.py`, `loc_infer.py`, `infer_mps.py`: real-time
  detection and localization; the last uses Core ML.
- `vision/location.py`: the localization class called by our control program.
- `vision/calibration/`: intrinsic/extrinsic calibration, homography, the
  ChArUco board generator and the calibration configs from our experiments.
- `vision/gen_yolo_data.py`, `SAMlabel_interactive.py`,
  `gen_split_dataset.py`: image collection, assisted labeling and training-set
  splitting.
- `vision/SAM_versions/`: our earlier labeling versions.
- `EdgeTAM/`: source, configs, install files and license of the on-device
  implementation.
- `SAM3/`: the SAM3 model and training source we used, with configs, the
  tokenizer vocabulary and our custom labeling entrypoint `main.py`.

## SAM3 and on-device deployment

We adopted EdgeTAM from the upstream EdgeTAM source as our on-device
implementation, to cover more devices and deployment scenarios. This package
ships the actual source of both SAM3 and EdgeTAM; the upstream authors and
licenses are kept in their respective directories.

`SAM3/main.py` is our existing entrypoint for propagation labeling with YOLO
box prompts on the first N frames. Following the upstream requirements, SAM3
uses Python 3.12, PyTorch >= 2.7 and CUDA >= 12.6; install SAM3 and EdgeTAM in
separate environments to avoid dependency conflicts. Inside `SAM3/`:

```bash
python -m pip install -e ".[train,notebooks]"
python main.py --img_dir /path/to/frames --label_dir /path/to/labels --checkpoint /path/to/sam3.pt --gpus 0
```

Install a matching CUDA PyTorch build yourself and provide the data and
weights. The training entrypoint is `sam3/train/train.py`; see
`SAM3/README_TRAIN.md` for the data format and configs.

## Installation and training

Use Python 3.10 or newer. From this directory:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e EdgeTAM
python vision/yolo/train.py --help
python vision/yolo/train.py --data /path/to/micro.yaml --model /path/to/yolo11m.pt --device cpu
```

Change `--device cpu` to `mps` or a CUDA device index according to your
hardware. `vision/yolo/micro.yaml` is the data config template: set `path` to
the absolute path of a dataset directory containing `images/train`,
`images/val` and the corresponding `labels`. The class is `robot`.

## Collection, labeling and localization

Run all of the following from this directory. Provide camera intrinsics and
extrinsics that match your setup; the bundled YAML files are the calibration
from our experiments, so recalibrate after changing the camera or the plane,
and place the generated YAML back under `vision/calibration/configs/`.

```bash
python vision/gen_yolo_data.py
python vision/SAMlabel_interactive.py
python vision/gen_split_dataset.py
python vision/yolo/infer_base.py
```

To decode videos directly with EdgeTAM, also install `requirements-video.txt`;
the interactive labeling reads image sequences by default.

Labeling requires your own `EdgeTAM/checkpoints/edgetam.pt`. Real-time
detection expects `vision/yolo/models/best.pt`, and the localization class uses
`best_s.pt`. The Mac Core ML entrypoint uses `best.mlpackage`: install
`requirements-coreml.txt` first, then run `python vision/yolo/infer_mps.py`.
