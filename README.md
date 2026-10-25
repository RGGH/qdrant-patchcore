# patchcore-qdrant

Minimal **few-shot anomaly detection (FSAD)** in Rust: [PatchCore](https://arxiv.org/abs/2106.08265) with [Qdrant](https://qdrant.tech) as the memory bank.

Show it a handful of defect-free images, and it flags anything that looks different — no defect examples, no training.

## How it works

```
 normal images ──► CNN patch features ──► (coreset) ──► Qdrant collection   (fit)
                        ▲                                    │
 test image ────────────┘──► 1-NN distance per patch ◄───────┘             (score)
                                    │
                    image score = max patch distance   +   anomaly map
```

1. **Features** – WideResNet-50 layer2 + layer3, 3×3 local aggregation, concatenated and reduced to 384-d. The extractor runs from Rust through ONNX Runtime (`export_onnx.py` produces the model). A 224×224 image gives 28×28 = 784 patch vectors.
2. **Memory bank** – patch vectors of the few normal images are stored in a Qdrant collection with Euclidean distance. Optional greedy k-center coreset (`--coreset`) shrinks the bank.
3. **Scoring** – each test patch asks Qdrant for its nearest normal patch. The largest distance is the image's anomaly score; all distances form the anomaly map.

## What `backbone.onnx` is

It plays the role that fastembed or CLIP would play in other vector-search projects: it turns images into vectors that Qdrant can search. The difference is granularity.

- **fastembed / CLIP** give one vector per whole image, trained so that images with similar meaning land close together. One vector per image blends everything together, so a small defect would barely move it.
- **`backbone.onnx`** gives 784 vectors per image, one per small patch. They come from the middle layers (layer2 and layer3) of a network, which capture local texture and structure, such as a nut's surface or the edge of a crack. Qdrant then compares patch to patch.

**How it is made.** `export_onnx.py` uses PyTorch and torchvision:

1. Torchvision loads a WideResNet-50 with its standard ImageNet-trained weights.
2. The script wraps it with the extra PatchCore steps: local 3×3 averaging, upsampling the coarser layer, joining the two layers, and reducing the result to 384 numbers.
3. `torch.onnx.export` traces all of that and saves it as `backbone.onnx`.

Torch is only needed for this one export. At run time the Rust program loads the file with the `ort` crate (ONNX Runtime) and never touches Python. fastembed also runs its models through ONNX Runtime, so the underlying approach is the same.

## Requirements

- Rust (stable) and a network connection for the first build (the `ort` crate downloads ONNX Runtime)
- Python 3 with `torch>=2.5` and `torchvision` (only to export the model once)
- Docker (or any other way to run Qdrant)

## Quick start

```bash
# 1. Start Qdrant (REST :6333, gRPC :6334)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 2. Export the feature extractor once -> backbone.onnx (in the project root)
pip install torch torchvision onnx
python3 export_onnx.py        # run as a script, not `python3 -m export_onnx.py`

# 3. Build the memory bank from a few normal images
#    (MVTec AD category, here "hazelnut": data/train/good, data/test/<defect>)
cargo run --release -- --collection hazelnut fit data/train/good --shots 8

# 4. Score test images (add --save-map for <image>.map.png heat maps)
cargo run --release -- --collection hazelnut score \
    data/test/good/*.png data/test/crack/*.png data/test/cut/*.png \
    data/test/hole/*.png data/test/print/*.png
```

The export downloads the pretrained WideResNet-50 weights on first run (cached under `~/.cache/torch`). A `DeprecationWarning` about the legacy ONNX exporter is harmless; success ends with `wrote backbone.onnx`.

Example output (the shape of it; your numbers will differ):

```
   1.912  data/test/good/000.png
   4.376  data/test/crack/000.png
```

## Commands

```
patchcore-qdrant [--qdrant URL] [--collection NAME] [--model backbone.onnx] <COMMAND>

fit   <DIR>     [--shots N] [--coreset RATIO]   build / replace the memory bank
score <IMAGE>…  [--threshold T] [--save-map]    anomaly score per image
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--qdrant` | `http://localhost:6334` | Qdrant **gRPC** endpoint |
| `--collection` | `patchcore` | One collection per object category |
| `--model` | `backbone.onnx` | ONNX feature extractor |
| `--shots` | all | Use only the first N images of `DIR` |
| `--coreset` | `1.0` | Fraction of patches kept (e.g. `0.1`); with only a few shots, keeping everything is usually best |
| `--threshold` | – | Print `ANOMALOUS` / `normal` using this score cut-off |
| `--save-map` | off | Write a min-max normalised anomaly map next to each image |

## Choosing a threshold

Scores are only comparable within one collection. Score a few **held-out normal** images (not used in `fit`) and set the threshold slightly above their highest score, then pass it with `--threshold`:

```bash
cargo run --release -- --collection hazelnut score data/test/good/*.png | sort -n | tail -1
cargo run --release -- --collection hazelnut score data/test/crack/*.png --threshold <that score> --save-map
```

`--save-map` writes `<name>.map.png` next to each image, i.e. inside `data/test/...`.

## Troubleshooting

- `File at backbone.onnx does not exist` – run `python3 export_onnx.py` first, from the directory you run `cargo` in (or pass `--model path/to/backbone.onnx`).
- Connection errors – Qdrant must be running and reachable on the **gRPC** port (6334), not only 6333.
- Good and defective scores overlap – use more shots (`--shots 16`, `32`). Some MVTec categories (e.g. hazelnut) contain randomly rotated objects, and this pipeline has no rotation handling, so the bank needs to see more orientations.

## Notes and limitations

- Fixed 224×224 input, no centre crop; objects should fill the frame and be roughly aligned (as in MVTec AD).
- The reduction to 384-d is a simple channel average, in the spirit of PatchCore's adaptive average pooling. Change `--dim` in `export_onnx.py` (must divide 1536); the Rust side adapts automatically.
- Few-shot banks are small, so Qdrant answers with an exact scan (its default below 10,000 vectors). Larger banks switch to the HNSW index: faster, with a small recall cost.
- `fit` deletes and recreates the collection each time.
- If the ONNX Runtime download fails (offline or restricted network), build with the `load-dynamic` feature of `ort` and set `ORT_DYLIB_PATH` to a local `libonnxruntime` (for example the one shipped inside the `onnxruntime` pip package).


## Images from Kaggle

```bash
#!/bin/bash
curl -L -o ~/Downloads/mvtec-ad.zip\
  https://www.kaggle.com/api/v1/datasets/download/ipythonx/mvtec-ad
```

## Successful output example
```bash
❯ cargo run --release -- --collection hazelnut score data/test/print/*.png --threshold 0.77 --save-map
    Finished `release` profile [optimized] target(s) in 0.28s
     Running `target/release/patchcore-qdrant --collection hazelnut score data/test/print/000.png data/test/print/001.png data/test/print/002.png data/test/print/003.png data/test/print/004.png data/test/print/005.png data/test/print/006.png data/test/print/007.png data/test/print/008.png data/test/print/009.png data/test/print/010.png data/test/print/011.png data/test/print/012.png data/test/print/013.png data/test/print/014.png data/test/print/015.png data/test/print/016.png --threshold 0.77 --save-map`
   1.015  data/test/print/000.png  ANOMALOUS
   0.984  data/test/print/001.png  ANOMALOUS
   1.122  data/test/print/002.png  ANOMALOUS
   0.856  data/test/print/003.png  ANOMALOUS
   0.954  data/test/print/004.png  ANOMALOUS
   0.903  data/test/print/005.png  ANOMALOUS
   1.036  data/test/print/006.png  ANOMALOUS
   0.972  data/test/print/007.png  ANOMALOUS
   1.232  data/test/print/008.png  ANOMALOUS
   0.884  data/test/print/009.png  ANOMALOUS
   1.043  data/test/print/010.png  ANOMALOUS
   1.027  data/test/print/011.png  ANOMALOUS
   1.040  data/test/print/012.png  ANOMALOUS
   0.859  data/test/print/013.png  ANOMALOUS
   0.894  data/test/print/014.png  ANOMALOUS
   0.775  data/test/print/015.png  ANOMALOUS
   0.913  data/test/print/016.png  ANOMALOUS
```

## Files

```
Cargo.toml        dependencies
src/main.rs       fit / score pipeline (~210 lines)
export_onnx.py    one-off ONNX export of the feature extractor
```
