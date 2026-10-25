# Why Qdrant Works Well with PatchCore

## What backbone.onnx does 

`backbone.onnx` is the system's eyes: a frozen, pretrained image-recognition network that turns each small patch of an image into a numeric fingerprint.

It is a WideResNet-50 trained on ImageNet and saved in ONNX, a portable format that Rust can run without Python. It does not learn anything from your data.

It splits a 224×224 image into a 28×28 grid of patches (784 in total) and turns each patch into a list of 384 numbers. Patches that look alike get similar fingerprints: a normal hazelnut surface gives familiar fingerprints, and a crack gives an unfamiliar one.

Qdrant stores the fingerprints of normal patches, and scoring measures how far each new fingerprint is from its closest normal one.

## Why Qdrant fits

PatchCore's memory bank is a set of patch vectors searched by nearest neighbour, which is exactly what a vector database does.

- **Persistence:** run `fit` once, and `score` can run later from any process without recomputing features.
- **Scales up:** small banks get an exact scan (below 10,000 vectors by default). Larger ones switch to an HNSW index, which is fast at a small recall cost.
- **Native Euclidean search:** PatchCore uses L2 distance, and Qdrant returns that distance directly.
- **One collection per category:** `hazelnut`, `bottle` and so on sit side by side without affecting each other.
- **Production features:** network API, authentication, snapshots, replication, on-disk storage.

## Sharing across production-line machines

Qdrant can let several machines share one memory bank: any machine that can reach the server can score against the same collection, so every station compares against the same normal patches.

Only the normal patch vectors are shared. A test image's patches are computed locally and sent as queries, and they are not stored.

Caveats:

- **Same extractor everywhere:** every machine needs the identical `backbone.onnx` and preprocessing. Otherwise vectors are not comparable and scores silently become meaningless.
- **Don't re-run `fit` live:** it deletes and recreates the collection. Build a new one (`hazelnut_v2`) and switch machines over.
- **Network cost:** each image sends about 1.2 MB of query vectors, and the server is a single point of failure unless replicated.
- **Security not set up:** the tool uses a plain URL. A real line needs an API key and TLS, which needs a small code change.

I have not tested multi-machine use. It follows from how Qdrant works.

## Trade-offs

For tiny few-shot banks, Qdrant is more machinery than you strictly need, since an in-memory array with brute-force search would also work.

It adds a service to run and a network round trip per query. Features and the coreset step still run in Rust, and only the search happens in Qdrant.

Qdrant matters most with many categories, large banks, or a shared production deployment, and matters less for a small experiment.
