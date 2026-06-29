#!/usr/bin/env bash
# PKG-1 — package the `primitive` macOS app.
#
# Part A (this script, autonomous, NO credentials, NO network): build the .app with cargo-bundle and
# validate it, then **HALT immediately before the first codesign step**, printing the exact Part B
# commands. This script NEVER calls codesign / notarytool / stapler itself — those touch Apple
# credentials and the network and MUST be run interactively, with you in the loop (plan §7 PKG-1).
#
# Part B (you run the printed commands by hand, confirming each step): codesign → create-dmg →
# notarytool submit → stapler staple. See the HALT message below for the literal commands.
#
# Usage: scripts/ops/sign-notarize.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Prefer the rustup toolchain (Homebrew's cargo shadows it and can't build the pinned deps).
export PATH="$HOME/.cargo/bin:$PATH"

APP_NAME="primitive"
BUNDLE_ID="com.primitive.app"
APP_PATH="target/release/bundle/osx/${APP_NAME}.app"

echo "==> [Part A] Validating canonical Info.plist"
plutil -lint assets/Info.plist

echo "==> [Part A] Building the app bundle (cargo bundle --release)"
if ! command -v cargo-bundle >/dev/null 2>&1; then
  echo "ERROR: cargo-bundle not installed. Run: cargo install cargo-bundle" >&2
  exit 1
fi
# cargo-bundle reads [package.metadata.bundle] from the *package* manifest, so run it from the
# crate dir (the virtual workspace root has no [package]); output still lands in the shared target/.
# `--format osx` → just the unsigned .app; the DMG is built in Part B *from the signed app* (the
# canonical order is sign → package → notarize, so we don't ship the unsigned dmg cargo would make).
( cd crates/primitive-app && cargo bundle --release --format osx )

if [ ! -d "${APP_PATH}" ]; then
  echo "ERROR: expected bundle not found at ${APP_PATH}" >&2
  exit 1
fi

echo "==> [Part A] Validating the produced bundle"
plutil -lint "${APP_PATH}/Contents/Info.plist"
test -x "${APP_PATH}/Contents/MacOS/${APP_NAME}" || {
  echo "ERROR: bundle executable missing/not executable" >&2
  exit 1
}
echo "    OK — ${APP_PATH} built and structurally valid."

cat <<HALT

────────────────────────────────────────────────────────────────────────────
HALT: stopping BEFORE codesign. Part A is complete and touched NO credentials.

Part B is interactive — run each command yourself, confirming as you go. You
hold the Apple credentials; this script will not run these for you.

  # 1. Sign (Developer ID Application cert must be in your login keychain)
  codesign --sign "Developer ID Application: <YOUR NAME> (<TEAMID>)" \\
    --timestamp --options runtime --deep --force "${APP_PATH}"
  codesign --verify --strict --verbose=2 "${APP_PATH}"

  # 2. Build a DMG (brew install create-dmg)
  create-dmg --volname "${APP_NAME}" --app-drop-link 480 170 \\
    "target/release/bundle/osx/${APP_NAME}.dmg" "${APP_PATH}"

  # 3. Notarize (use an app-specific password or a notarytool keychain profile)
  xcrun notarytool submit "target/release/bundle/osx/${APP_NAME}.dmg" \\
    --apple-id "<APPLE_ID_EMAIL>" --team-id "<TEAMID>" \\
    --password "<APP_SPECIFIC_PASSWORD>" --wait

  # 4. Staple the ticket onto the .app (and/or the .dmg)
  xcrun stapler staple "${APP_PATH}"

  # 5. Verify on a CLEAN machine (no dev tools) — the real done-gate (plan §7 PKG-1):
  xcrun stapler validate "${APP_PATH}"          # → "The validate action worked!"
  spctl --assess --type execute -vv "${APP_PATH}"  # → "accepted ... source=Notarized Developer ID"
  #   bundle id should read: ${BUNDLE_ID}
────────────────────────────────────────────────────────────────────────────
HALT
