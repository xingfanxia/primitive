//! `primitive` — the GUI binary. All behavior lives in the [`primitive_app`] library (so the §5A
//! interaction logic, a11y math, device probe, and end-to-end run stay unit-testable headless).

fn main() -> eframe::Result<()> {
    primitive_app::run()
}
