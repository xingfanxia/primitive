//! String table (plan §5A localization: "all strings through a string table from day one; no
//! hard-coded UI text"). English first; adding a locale = another [`Strings`] constructor. The
//! renderer pulls every label from here so no user-facing literal is stranded in widget code.

/// Every user-facing string in the app, in one place.
#[derive(Clone, Copy, Debug)]
pub struct Strings {
    pub app_title: &'static str,
    pub source_header: &'static str,
    pub shapes_header: &'static str,
    pub advanced_header: &'static str,
    pub browse: &'static str,
    pub try_label: &'static str,
    pub type_label: &'static str,
    pub triangle: &'static str,
    pub ellipse: &'static str,
    pub rectangle: &'static str,
    pub count: &'static str,
    pub alpha: &'static str,
    pub seed: &'static str,
    pub restarts_n: &'static str,
    pub age_m: &'static str,
    pub attempts: &'static str,
    pub reduce_motion: &'static str,
    pub start: &'static str,
    pub pause: &'static str,
    pub resume: &'static str,
    pub reset: &'static str,
    pub export: &'static str,
    pub export_png: &'static str,
    pub export_svg: &'static str,
    pub export_gif: &'static str,
    pub drop_to_begin: &'static str,
    pub chip_tooltip: &'static str,
    pub err_read_image: &'static str,
    pub err_decode_sample: &'static str,
    pub err_save: &'static str,
    pub saved: &'static str,
    pub reveal_in_finder: &'static str,
}

impl Strings {
    /// English (the default locale).
    pub const fn en() -> Strings {
        Strings {
            app_title: "primitive",
            source_header: "SOURCE",
            shapes_header: "SHAPES",
            advanced_header: "Advanced",
            browse: "Drop image · or Browse…",
            try_label: "try:",
            type_label: "type",
            triangle: "△ triangle",
            ellipse: "◯ ellipse",
            rectangle: "▭ rect",
            count: "count",
            alpha: "alpha",
            seed: "seed",
            restarts_n: "restarts (n)",
            age_m: "age",
            attempts: "attempts (m)",
            reduce_motion: "Reduce motion",
            start: "▶ Start",
            pause: "⏸ Pause",
            resume: "▶ Resume",
            reset: "↺ Reset",
            export: "⤓ Export",
            export_png: "PNG",
            export_svg: "SVG",
            export_gif: "GIF",
            drop_to_begin: "Drop an image to begin",
            chip_tooltip: "Live preview runs on the CPU adapter (the parity oracle). \
                A detected GPU enables an instant (non-streamed) run.",
            err_read_image: "Couldn't read that image — try PNG, JPG, or WebP",
            err_decode_sample: "Couldn't decode sample",
            err_save: "Couldn't save — check folder permissions",
            saved: "Saved",
            reveal_in_finder: "Reveal in Finder",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_string_is_empty() {
        // A cheap guard that every table slot is filled (a stranded "" would render as a blank
        // label). We check the load-bearing ones explicitly.
        let s = Strings::en();
        for v in [
            s.app_title,
            s.source_header,
            s.start,
            s.export,
            s.drop_to_begin,
            s.reduce_motion,
            s.advanced_header,
        ] {
            assert!(!v.is_empty(), "string table slot must not be empty");
        }
    }
}
