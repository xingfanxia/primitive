//! CORE-1 golden-image gate.
//!
//! Two assertions, per plan §7 CORE-1 ("output matches a fogleman reference at SSIM ≥ 0.999
//! and final SSE within 1%"):
//!
//! 1. **Determinism / regression** — a fixed seed + fixed target produce an output that
//!    matches the committed golden at **SSIM ≥ 0.999**. The golden is this port's own
//!    fixed-seed reconstruction (a true fogleman *image* match is impossible: fogleman seeds
//!    with wall-clock time and is non-reproducible run-to-run). The math equivalence to
//!    fogleman is proven separately and exactly in `parity.rs`.
//! 2. **Quality vs fogleman** — the final normalized-RMSE score is within **1%** of an actual
//!    fogleman run on the *byte-identical* resized target.
//!
//! The SSIM tolerance (≥ 0.999, not == 1.0) absorbs sub-ULP libm differences across machines
//! while still catching any real algorithm/math regression.

mod common;

use common::{canvas_to_image, resize_target_canvas, ssim};
use primitive_core::{Model, Rng, ShapeType};

const TARGET_IMAGE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/picasso.jpg");
/// The committed resized target fogleman optimized — provenance for `FOGLEMAN_REF_SCORE`.
const TARGET_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/target_picasso_128.png"
);
const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/golden_picasso_128_t100.png"
);
const SIZE: u32 = 128;
const COUNT: i32 = 100;
const SEED: u64 = 1;

/// fogleman's final score on the **committed** byte-identical target (`TARGET_FIXTURE`,
/// `-r 0 -s 128 -n 100 -m 1 -j 1`), sampled over 5 wall-clock-seeded runs on 2026-06-27:
///   0.062803, 0.063347, 0.062061, 0.062870, 0.062208  →  mean 0.062658.
/// Reproduce with `scripts/ops/regen-fogleman-ref.sh`. This port's fixed-seed score must be
/// within 1% of this reference.
const FOGLEMAN_REF_SCORE: f64 = 0.062658;

fn run_reference() -> Model {
    let target = resize_target_canvas(TARGET_IMAGE, SIZE);
    let mut model = Model::with_average_background(target);
    let mut rng = Rng::new(SEED);
    model.run(ShapeType::Triangle, 128, COUNT, &mut rng);
    model
}

/// The committed target fixture must be byte-identical to this port's in-memory resize, so
/// "fogleman optimized the same pixels" is a checkable fact, not a claim.
#[test]
fn committed_target_matches_resize() {
    let in_memory = canvas_to_image(&resize_target_canvas(TARGET_IMAGE, SIZE));
    let committed = image::open(TARGET_FIXTURE)
        .expect("committed target fixture present")
        .to_rgba8();
    assert_eq!(in_memory.dimensions(), committed.dimensions());
    assert_eq!(
        in_memory.as_raw(),
        committed.as_raw(),
        "resize drifted from the committed target fogleman was scored against"
    );
}

/// Both CORE-1 assertions in one run (the search is the slow part — run it once).
#[test]
fn golden_ssim_and_quality() {
    let model = run_reference();

    // --- 1. determinism / regression (SSIM ≥ 0.999 vs committed golden) ---
    let out = canvas_to_image(&model.current);
    if !std::path::Path::new(GOLDEN).exists() {
        // A missing golden is a HARD FAILURE, never a silent self-certify: regenerating
        // after a regression would otherwise bake the regression into the new golden.
        // Mirrors parity.rs, which hard-fails on a missing fixture.
        if std::env::var_os("UPDATE_GOLDEN").is_some() {
            out.save(GOLDEN).expect("write golden fixture");
            panic!("created golden at {GOLDEN}; re-run without UPDATE_GOLDEN and commit it");
        }
        panic!("golden fixture missing: {GOLDEN} (regenerate with UPDATE_GOLDEN=1, then commit)");
    }
    let golden = image::open(GOLDEN).expect("open golden fixture").to_rgba8();
    let s = ssim(&out, &golden);
    println!("golden SSIM = {s:.6}");
    assert!(s >= 0.999, "golden SSIM {s:.6} < 0.999 — output regressed");

    // --- 2. quality within 1% of fogleman (provenance: scripts/ops/regen-fogleman-ref.sh) ---
    let rel = (model.score - FOGLEMAN_REF_SCORE).abs() / FOGLEMAN_REF_SCORE;
    println!(
        "final_score={:.6} fogleman_ref={:.6} relative_diff={:.4}% psnr={:.3}dB",
        model.score,
        FOGLEMAN_REF_SCORE,
        rel * 100.0,
        model.psnr()
    );
    assert!(
        rel <= 0.01,
        "final score {:.6} differs from fogleman by {:.3}% (> 1%)",
        model.score,
        rel * 100.0
    );
}
