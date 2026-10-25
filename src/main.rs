//! Few-shot anomaly detection (FSAD) with PatchCore, using Qdrant as the memory bank.
//!
//!   fit   : normal images -> patch features -> (optional coreset) -> Qdrant collection
//!   score : image -> patch features -> 1-NN distance per patch via Qdrant -> anomaly score + map

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use image::{imageops::FilterType, GrayImage, Luma};
use ndarray::{Array2, Array4};
use ort::{session::Session, value::Tensor};
use qdrant_client::{
    qdrant::{
        CreateCollectionBuilder, DeleteCollectionBuilder, Distance, PointStruct,
        QueryBatchPointsBuilder, QueryPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
    },
    Payload, Qdrant,
};

const SIZE: u32 = 224; // input resolution -> 28x28 = 784 patches
const MEAN: [f32; 3] = [0.485, 0.456, 0.406]; // ImageNet statistics
const STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Parser)]
struct Cli {
    /// Qdrant gRPC endpoint
    #[arg(long, default_value = "http://localhost:6334", global = true)]
    qdrant: String,
    /// Collection name (one memory bank per object category)
    #[arg(long, default_value = "patchcore", global = true)]
    collection: String,
    /// ONNX feature extractor produced by export_onnx.py
    #[arg(long, default_value = "backbone.onnx", global = true)]
    model: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build the memory bank from a folder of normal (defect-free) images
    Fit {
        dir: PathBuf,
        /// Use only the first N images (the "few shots")
        #[arg(long)]
        shots: Option<usize>,
        /// Fraction of patches kept by greedy coreset subsampling (1.0 = keep all)
        #[arg(long, default_value_t = 1.0)]
        coreset: f32,
    },
    /// Score images; higher = more anomalous
    Score {
        images: Vec<PathBuf>,
        /// Flag an image as anomalous above this score
        #[arg(long)]
        threshold: Option<f32>,
        /// Write a grayscale anomaly map next to each image (<name>.map.png)
        #[arg(long)]
        save_map: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = Qdrant::from_url(&cli.qdrant).build()?;
    let env = ort::init().build()?;
    let mut session = Session::builder(&env)?.commit_from_file(&cli.model)?;

    match cli.cmd {
        Cmd::Fit { dir, shots, coreset } => {
            let mut files = list_images(&dir)?;
            files.truncate(shots.unwrap_or(usize::MAX));
            if files.is_empty() {
                bail!("no images found in {}", dir.display());
            }

            // 1. Patch features of every normal image, stacked into one [N, D] matrix.
            let mut bank: Vec<Array2<f32>> = Vec::new();
            for f in &files {
                bank.push(extract(&mut session, f)?);
            }
            let views: Vec<_> = bank.iter().map(|a| a.view()).collect();
            let bank = ndarray::concatenate(ndarray::Axis(0), &views)?;

            // 2. Optional greedy k-center coreset.
            let keep = greedy_coreset(&bank, ((bank.nrows() as f32) * coreset).ceil() as usize);
            println!("{} images -> {} patches -> {} kept", files.len(), bank.nrows(), keep.len());

            // 3. Store in Qdrant. Euclidean distance == PatchCore's L2 nearest neighbour.
            if client.collection_exists(&cli.collection).await? {
                client.delete_collection(DeleteCollectionBuilder::new(&cli.collection)).await?;
            }
            client
                .create_collection(
                    CreateCollectionBuilder::new(&cli.collection)
                        .vectors_config(VectorParamsBuilder::new(bank.ncols() as u64, Distance::Euclid)),
                )
                .await?;
            for chunk in keep.chunks(500) {
                let points: Vec<_> = chunk
                    .iter()
                    .map(|&i| PointStruct::new(i as u64, bank.row(i).to_vec(), Payload::new()))
                    .collect();
                client
                    .upsert_points(UpsertPointsBuilder::new(&cli.collection, points).wait(true))
                    .await?;
            }
            println!("memory bank stored in collection '{}'", cli.collection);
        }

        Cmd::Score { images, threshold, save_map } => {
            for path in images {
                let patches = extract(&mut session, &path)?;

                // 1-NN distance of every test patch to the memory bank.
                let mut dist = Vec::with_capacity(patches.nrows());
                for rows in (0..patches.nrows()).collect::<Vec<_>>().chunks(256) {
                    let queries: Vec<_> = rows
                        .iter()
                        .map(|&i| {
                            QueryPointsBuilder::new(&cli.collection)
                                .query(patches.row(i).to_vec())
                                .limit(1)
                                .build()
                        })
                        .collect();
                    let res = client
                        .query_batch(QueryBatchPointsBuilder::new(&cli.collection, queries))
                        .await?;
                    dist.extend(res.result.iter().map(|r| r.result[0].score));
                }

                // Image-level score = worst patch.
                let score = dist.iter().cloned().fold(f32::MIN, f32::max);
                let verdict = match threshold {
                    Some(t) if score > t => "  ANOMALOUS",
                    Some(_) => "  normal",
                    None => "",
                };
                println!("{:8.3}  {}{}", score, path.display(), verdict);

                if save_map {
                    let out = path.with_extension("map.png");
                    anomaly_map(&dist).save(&out)?;
                }
            }
        }
    }
    Ok(())
}

/// Image -> [num_patches, dim] patch descriptors (the ONNX model does the CNN +
/// neighbourhood aggregation + dimensionality reduction).
fn extract(session: &mut Session, path: &Path) -> Result<Array2<f32>> {
    let img = image::open(path)?
        .resize_exact(SIZE, SIZE, FilterType::Triangle)
        .to_rgb8();
    let mut x = Array4::<f32>::zeros((1, 3, SIZE as usize, SIZE as usize));
    for (px, py, p) in img.enumerate_pixels() {
        for c in 0..3 {
            x[[0, c, py as usize, px as usize]] = (p[c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }
    let out = session.run(ort::inputs!["image" => Tensor::from_array(x)?])?;
    let (shape, data) = out["patches"].try_extract_tensor::<f32>()?;
    Ok(Array2::from_shape_vec((shape[1] as usize, shape[2] as usize), data.to_vec())?)
}

/// Greedy k-center coreset: repeatedly pick the patch farthest from everything chosen so far.
fn greedy_coreset(feats: &Array2<f32>, k: usize) -> Vec<usize> {
    let n = feats.nrows();
    if k >= n {
        return (0..n).collect();
    }
    let mut chosen = vec![0usize];
    let mut min_d = vec![f32::MAX; n];
    while chosen.len() < k {
        let c = feats.row(*chosen.last().unwrap());
        for (i, row) in feats.rows().into_iter().enumerate() {
            let d: f32 = row.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
            min_d[i] = min_d[i].min(d);
        }
        let next = (0..n).max_by(|&a, &b| min_d[a].total_cmp(&min_d[b])).unwrap();
        chosen.push(next);
    }
    chosen
}

/// Per-patch distances -> min-max normalised grayscale heat map at input resolution.
fn anomaly_map(dist: &[f32]) -> GrayImage {
    let side = (dist.len() as f32).sqrt() as u32;
    let (lo, hi) = dist.iter().fold((f32::MAX, f32::MIN), |(l, h), &d| (l.min(d), h.max(d)));
    let small = GrayImage::from_fn(side, side, |x, y| {
        let d = dist[(y * side + x) as usize];
        Luma([(255.0 * (d - lo) / (hi - lo).max(1e-6)) as u8])
    });
    image::imageops::resize(&small, SIZE, SIZE, FilterType::Triangle)
}

fn list_images(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()).map(str::to_lowercase).as_deref(),
                Some("png" | "jpg" | "jpeg")
            )
        })
        .collect();
    v.sort();
    Ok(v)
}
