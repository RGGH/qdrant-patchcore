# Results explained

Output of `score` on the MVTec AD hazelnut test set, using a memory bank built from 8 normal training images (`fit data/train/good --shots 8`).

## How to read the output

Each line is `anomaly score  image path`. The score is the distance from the single most unusual patch in the test image to its closest match among the normal patches stored in Qdrant. Higher means more anomalous. There is no unit or fixed scale, so only compare scores from this one collection.

The score is a Euclidean distance in the 384-dimensional feature space, not a probability, so it is not limited to the 0 to 1 range. Scores above 1 are normal: 9 of the 110 test images exceeded 1, and all 9 are defects (`hole/003`, `hole/007`, and `print/000`, `002`, `006`, `008`, `010`, `011`, `012`, with `print/008` the highest at 1.232). No good, crack or cut image went above 1.

## Getting a verdict instead of just a score

Without `--threshold`, the tool prints only the score and the path for each image. It does not say whether an image is good or defective; that call is left to you.

To get a verdict, add a threshold:

```bash
cargo run --release -- --collection hazelnut score data/test/print/*.png --threshold 0.77
```

Each line then ends with `ANOMALOUS` (score above 0.77) or `normal`, for example:

```
   0.775  data/test/print/015.png  ANOMALOUS
```

Add `--save-map` if you also want a heat map image written next to each test image, showing where the strongest anomalies are.

## What the scores show

| Folder | Images | Score range | Typical |
| --- | --- | --- | --- |
| good | 40 | 0.52 – 0.76 | ~0.65 |
| crack | 18 | 0.69 – 0.93 | ~0.81 |
| cut | 17 | 0.74 – 0.86 | ~0.80 |
| hole | 18 | 0.75 – 1.10 | ~0.93 |
| print | 17 | 0.78 – 1.23 | ~0.97 |

It works: every defect type scores higher than good on average, and the order makes sense. Print and hole are the easiest to catch because they are large and visibly different. Crack and cut are subtler, so they sit closer to the good scores.

The gap is modest rather than dramatic, about 0.65 for good against 0.8 to 1.0 for defects. Even good images score well above zero because the bank only holds 8 training images, so some natural variation in a good nut was never seen. More shots should pull the good scores down.

## Choosing a threshold

The ranges overlap a little at the edges. For example, `crack/012` (0.689) scores lower than 28 of the 40 good images. If you set a threshold just above the highest good score (0.761), say 0.77, you get:

- **Good images:** none flagged, by construction.
- **Defects:** 8 of 70 missed (4 crack, 3 cut, 1 hole), so 62 of 70 caught.
- **Per type:** print is fully caught, and the misses are the lowest-scoring defects in the table above.

Do not treat 1.0 as a natural cutoff just because it is a round number. A threshold of 1.0 would catch only those 9 of 70 defects, while 0.77 catches 62. Good images top out at 0.761, so the useful threshold is just above that.

That threshold was chosen using the same 40 good images, so the zero false positives is optimistic. To judge it fairly, pick the threshold from some good images and test on the rest. A single maximum is also fragile, since one unusual good nut would move it.

## Improving separation

- Try `--shots 16` or `32`. Hazelnuts are randomly rotated and this pipeline has no rotation handling, so the bank needs to see more orientations.
- For a single threshold-free quality number, an `eval` command reporting AUROC across the good and defect folders could be added.
