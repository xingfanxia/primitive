#!/usr/bin/env bash
# PKG-1 — package the `primitive` macOS app.
#
# DEFAULT (no args) — Part A: build the .app with cargo-bundle + validate it, then HALT before any
#   signing. Autonomous, NO credentials, NO network. This is what `make bundle` runs.
#
# WITH --sign — Part B: the full Developer ID distribution chain (codesign → notarize → staple →
#   verify) using YOUR Apple credentials, read from the environment. Interactive/credentialed — you
#   run it. Nothing signs, uploads, or touches the network unless you pass --sign.
#
# Part B credentials (environment variables):
#   SIGN_IDENTITY   "Developer ID Application: NAME (TEAMID)"      (required; see `security find-identity -v -p codesigning`)
#   and ONE notarization credential set:
#     NOTARY_PROFILE   an xcrun notarytool keychain-profile name   (recommended), OR
#     APPLE_ID + TEAM_ID + APP_PASSWORD                            (app-specific password, appleid.apple.com)
#
# One-time setup, the app-specific password, and troubleshooting: docs/pkg/RUNBOOK.md
#
# Usage:
#   scripts/ops/sign-notarize.sh                                   # Part A only (build + validate + halt)
#   SIGN_IDENTITY="Developer ID Application: Jane Doe (ABCDE12345)" \
#     NOTARY_PROFILE=primitive-notary \
#     scripts/ops/sign-notarize.sh --sign                          # Part A + Part B (sign → notarize → staple)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Prefer the rustup toolchain (Homebrew's cargo shadows it and can't build the pinned deps).
export PATH="$HOME/.cargo/bin:$PATH"

APP_NAME="primitive"
BUNDLE_ID="com.primitive.app"
DIST_DIR="target/release/bundle/osx"
APP_PATH="${DIST_DIR}/${APP_NAME}.app"
DMG_PATH="${DIST_DIR}/${APP_NAME}.dmg"
ZIP_PATH="${DIST_DIR}/${APP_NAME}.zip"

MODE="parta"
case "${1:-}" in
  "")        MODE="parta" ;;
  --sign)    MODE="sign" ;;
  -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
  *) echo "ERROR: unknown argument '$1' (expected --sign or no args)" >&2; exit 2 ;;
esac

# ─────────────────────────────────────────────────────────────────────────────
# Part A — build + validate the unsigned bundle (no credentials, no network).
# ─────────────────────────────────────────────────────────────────────────────
part_a() {
  echo "==> [Part A] Validating canonical Info.plist"
  plutil -lint assets/Info.plist

  echo "==> [Part A] Building the app bundle (cargo bundle --release)"
  if ! command -v cargo-bundle >/dev/null 2>&1; then
    echo "ERROR: cargo-bundle not installed. Run: cargo install cargo-bundle" >&2
    exit 1
  fi
  # cargo-bundle reads [package.metadata.bundle] from the *package* manifest, so run it from the crate
  # dir (the virtual workspace root has no [package]); output still lands in the shared target/.
  # `--format osx` → just the UNSIGNED .app; the DMG is built in Part B *from the signed app* (canonical
  # order is sign → package → notarize, so we never ship the unsigned dmg cargo would otherwise make).
  ( cd crates/primitive-app && cargo bundle --release --format osx )

  [ -d "${APP_PATH}" ] || { echo "ERROR: expected bundle not found at ${APP_PATH}" >&2; exit 1; }

  echo "==> [Part A] Validating the produced bundle"
  plutil -lint "${APP_PATH}/Contents/Info.plist"
  test -x "${APP_PATH}/Contents/MacOS/${APP_NAME}" || {
    echo "ERROR: bundle executable missing/not executable" >&2; exit 1; }
  echo "    OK — ${APP_PATH} built and structurally valid."
}

# ─────────────────────────────────────────────────────────────────────────────
# Part B — Developer ID: codesign → notarize → staple → verify (YOUR credentials).
# cargo-bundle produces an UNSIGNED bundle with NO hardened runtime, so we sign the whole chain here.
# ─────────────────────────────────────────────────────────────────────────────
part_b() {
  echo ""
  echo "==> [Part B] Developer ID sign + notarize + staple"

  # --- credentials preflight ---
  : "${SIGN_IDENTITY:?Set SIGN_IDENTITY=\"Developer ID Application: NAME (TEAMID)\" — see docs/pkg/RUNBOOK.md}"

  local -a NOTARY_AUTH
  if [ -n "${NOTARY_PROFILE:-}" ]; then
    NOTARY_AUTH=(--keychain-profile "$NOTARY_PROFILE")
  elif [ -n "${APPLE_ID:-}" ] && [ -n "${TEAM_ID:-}" ] && [ -n "${APP_PASSWORD:-}" ]; then
    NOTARY_AUTH=(--apple-id "$APPLE_ID" --team-id "$TEAM_ID" --password "$APP_PASSWORD")
  else
    echo "ERROR: set NOTARY_PROFILE (recommended), or APPLE_ID + TEAM_ID + APP_PASSWORD." >&2
    echo "       See docs/pkg/RUNBOOK.md → 'One-time setup'." >&2
    exit 1
  fi

  if ! security find-identity -v -p codesigning | grep -qF "$SIGN_IDENTITY"; then
    echo "ERROR: signing identity not found in your keychain:" >&2
    echo "         $SIGN_IDENTITY" >&2
    echo "       Available Developer ID identities:" >&2
    security find-identity -v -p codesigning >&2 || true
    exit 1
  fi

  local INNER="${APP_PATH}/Contents/MacOS/${APP_NAME}"

  # --- sign inside-out (NO --deep; deprecated for signing — TN2206): inner Mach-O first, then bundle.
  # Hardened runtime (--options runtime) + secure timestamp (--timestamp) are both REQUIRED to notarize.
  # No --entitlements: a plain wgpu/Metal + winit app needs none (Metal doesn't JIT; Rust is static).
  echo "==> [Part B] codesign (hardened runtime + secure timestamp, inside-out)"
  xattr -cr "$APP_PATH"   # strip Finder/quarantine detritus that breaks signing
  codesign --force --options runtime --timestamp --sign "$SIGN_IDENTITY" "$INNER"
  codesign --force --options runtime --timestamp --sign "$SIGN_IDENTITY" "$APP_PATH"

  echo "==> [Part B] verify signature (--deep IS correct on verify)"
  codesign --verify --deep --strict --verbose=2 "$APP_PATH"
  codesign -dvv "$APP_PATH" 2>&1 | grep -E 'flags|Timestamp' || true   # expect flags=0x10000(runtime) + a Timestamp

  # --- notarize the .app (zip is only transport; the ticket keys off the signature), then staple the app.
  echo "==> [Part B] notarize the app (notarytool submit --wait)"
  rm -f "$ZIP_PATH"
  ditto -c -k --keepParent "$APP_PATH" "$ZIP_PATH"   # ditto, not zip — plain zip mangles bundles
  xcrun notarytool submit "$ZIP_PATH" "${NOTARY_AUTH[@]}" --wait
  xcrun stapler staple "$APP_PATH"                   # embed the ticket in the on-disk .app

  # --- package the DMG FROM the stapled app, then sign + notarize + staple the DMG too.
  # (Notarizing the dmg does NOT ticket the inner app — staple both so each validates offline.)
  echo "==> [Part B] build + sign + notarize + staple the DMG"
  rm -f "$DMG_PATH"
  hdiutil create -volname "$APP_NAME" -srcfolder "$APP_PATH" -ov -format UDZO "$DMG_PATH"
  codesign --force --timestamp --sign "$SIGN_IDENTITY" "$DMG_PATH"   # container: no --options runtime
  xcrun notarytool submit "$DMG_PATH" "${NOTARY_AUTH[@]}" --wait
  xcrun stapler staple "$DMG_PATH"

  # --- final done-gate (plan §7 PKG-1) ---
  echo "==> [Part B] verify (the real done-gate)"
  xcrun stapler validate "$APP_PATH"                 # → "The validate action worked!"
  xcrun stapler validate "$DMG_PATH"
  spctl --assess --type execute -vv "$APP_PATH"      # → accepted … source=Notarized Developer ID

  echo ""
  echo "    ✅ signed + notarized + stapled (bundle id ${BUNDLE_ID}):"
  echo "         app: ${APP_PATH}"
  echo "         dmg: ${DMG_PATH}   ← distribute this"
}

print_halt() {
  cat <<HALT

────────────────────────────────────────────────────────────────────────────
HALT: Part A complete — touched NO credentials, NO network. The bundle is UNSIGNED.

To sign + notarize (Part B), set your credentials and re-run with --sign:

  SIGN_IDENTITY="Developer ID Application: <YOUR NAME> (<TEAMID>)" \\
    NOTARY_PROFILE=<your-notarytool-profile> \\
    scripts/ops/sign-notarize.sh --sign        # or:  make sign

One-time setup (Developer ID cert, app-specific password, notarytool profile),
the exact chain, and troubleshooting: docs/pkg/RUNBOOK.md
────────────────────────────────────────────────────────────────────────────
HALT
}

on_err() {
  echo "" >&2
  echo "✗ Part B failed. Common causes + fixes: docs/pkg/RUNBOOK.md → 'Troubleshooting'." >&2
}

part_a
if [ "$MODE" = "sign" ]; then
  trap on_err ERR
  part_b
else
  print_halt
fi
