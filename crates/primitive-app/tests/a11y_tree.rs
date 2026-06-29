//! GUI-2 gate 4 (plan architecture.md line 276): the AccessKit tree exposes the control labels and a
//! live-progress node, so VoiceOver can read them. Driven headless with egui_kittest — we load a
//! sample *through the real UI* (clicking the sample chip in the accessibility tree) and then assert
//! the tree, rather than poking app fields. Forces the CPU path so no GPU is probed in the test.

use eframe::egui;
use egui_kittest::kittest::Queryable;
use egui_kittest::Harness;

use primitive_app::app::PrimitiveApp;

#[test]
fn accesskit_tree_has_controls_and_live_progress() {
    // Avoid the GPU probe in `PrimitiveApp::new` (and keep the chip deterministic).
    // SAFETY: single-threaded test process.
    unsafe { std::env::set_var("PRIMITIVE_FORCE_CPU", "1") };

    let mut harness = Harness::builder()
        .with_size(egui::vec2(1040.0, 700.0))
        .build_eframe(|cc| PrimitiveApp::new(cc));

    // First frame: empty/first-run state. The sample chips are present and enabled.
    harness.run();
    harness.get_by_label("cat").click();
    // Process the click → loads the bundled sample (in-memory, no file dialog) → Ready/Staged.
    harness.run();

    // Control labels are in the tree (VoiceOver reads them). `query_all` so a label that legitimately
    // appears on more than one node (e.g. the "count" slider label + its SpinButton) doesn't panic.
    let has = |q: &str| harness.query_all_by_label_contains(q).next().is_some();
    assert!(has("Start"), "Start button label in the AccessKit tree");
    assert!(has("count"), "count control label in the AccessKit tree");
    assert!(has("alpha"), "alpha control label in the AccessKit tree");

    // The live-progress node exists even before Start (idle reads "0/<count>"). Default count = 250.
    assert!(
        has("0/250"),
        "AccessKit tree must expose a live-progress node",
    );

    unsafe { std::env::remove_var("PRIMITIVE_FORCE_CPU") };
}
