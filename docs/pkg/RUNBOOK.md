# PKG-1 Part B — sign, notarize & staple `primitive.app` (macOS Developer ID)

This is the interactive, credentialed half of PKG-1. Part A (`make bundle`) builds and validates an
**unsigned** `primitive.app` and halts. Part B turns that into a **notarized Developer ID** app +
DMG that run on other Macs without Gatekeeper blocking them. It uses **your** Apple credentials and
runs **only** when you invoke it with `--sign` — nothing here signs or touches the network otherwise.

`scripts/ops/sign-notarize.sh --sign` (or `make sign`) automates the whole chain below. This doc is
the one-time setup + what the script does + how to verify + troubleshooting.

## Prerequisites

- A **paid Apple Developer Program** membership (free accounts can't get a Developer ID cert or notarize).
- **Xcode Command Line Tools** (`xcode-select --install`) — provides `codesign`, `xcrun notarytool`, `stapler`.
- A **Developer ID Application** certificate installed in your **login keychain**. Create/download it from
  the Apple Developer portal (Certificates → “Developer ID Application”) and double-click the `.cer` to
  install (you must have its private key — generate the cert from a CSR made on this Mac, or import the
  `.p12`). Verify it's present:
  ```bash
  security find-identity -v -p codesigning
  #   1) ABC123…  "Developer ID Application: Jane Doe (ABCDE12345)"   ← copy this exact string
  ```
- `cargo install cargo-bundle` (Part A builds the `.app`).

## One-time setup

1. **App-specific password** (NOT your Apple ID password, NOT your portal login): create one at
   <https://appleid.apple.com> → *Sign-In & Security* → *App-Specific Passwords*.

2. **Store notary credentials in the keychain** (recommended — keeps the password out of your shell
   history; `store-credentials` validates it immediately):
   ```bash
   xcrun notarytool store-credentials "primitive-notary" \
     --apple-id "you@example.com" --team-id "ABCDE12345"
   # paste the app-specific password when prompted
   ```

## Sign + notarize + staple

Set the two variables and run — the script does the full chain:

```bash
export SIGN_IDENTITY="Developer ID Application: Jane Doe (ABCDE12345)"   # exact string from find-identity
export NOTARY_PROFILE="primitive-notary"                                  # the profile you stored above
make sign        # == scripts/ops/sign-notarize.sh --sign
```

Prefer not to store a profile? Skip `store-credentials` and pass the trio instead:
```bash
export SIGN_IDENTITY="Developer ID Application: Jane Doe (ABCDE12345)"
export APPLE_ID="you@example.com" TEAM_ID="ABCDE12345" APP_PASSWORD="abcd-efgh-ijkl-mnop"
make sign
```

What the script does, in order (the canonical Developer ID flow):

1. **Part A** — `cargo bundle` the unsigned `primitive.app`, validate its `Info.plist` + executable.
2. **codesign inside-out** — sign the inner Mach-O, then the `.app`, with `--options runtime`
   (hardened runtime, required to notarize) + `--timestamp` (secure timestamp, required). **No `--deep`**
   (Apple deprecated it for signing — it can seal nested code as a resource and get you rejected) and
   **no entitlements** (a plain wgpu/Metal + winit Rust app needs none).
3. **Verify the signature** — `codesign --verify --deep --strict` (here `--deep` *is* correct) and print
   `flags=0x10000(runtime)` + the `Timestamp`.
4. **Notarize the app** — `ditto`-zip it as transport, `xcrun notarytool submit … --wait`, then
   `stapler staple` the ticket onto the on-disk `.app`.
5. **DMG** — build a `.dmg` **from the stapled app**, sign it, notarize it, staple it (notarizing the dmg
   does *not* ticket the inner app — both are stapled so each validates offline).
6. **Done-gate verify** — `stapler validate` both, and `spctl --assess --type execute -vv` → expect
   `accepted … source=Notarized Developer ID`.

Distribute the resulting **`target/release/bundle/osx/primitive.dmg`**.

## Verify (what success looks like)

```bash
xcrun stapler validate target/release/bundle/osx/primitive.app   # → "The validate action worked!"
spctl --assess --type execute -vv target/release/bundle/osx/primitive.app
#   accepted
#   source=Notarized Developer ID
codesign -dvv target/release/bundle/osx/primitive.app 2>&1 | grep -E 'flags|Timestamp'
#   flags=0x10000(runtime)   +   Timestamp=<a real date>
```

## Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `The executable does not have the hardened runtime enabled` | a Mach-O was signed without `--options runtime` → re-run `make sign` (it signs every binary with it) |
| `The signature does not include a secure timestamp` | `--timestamp` missing or `timestamp.apple.com` unreachable → ensure network, re-run |
| `not signed with a valid Developer ID certificate` | you used an "Apple Development" cert → set `SIGN_IDENTITY` to the **Developer ID Application** line from `security find-identity -v -p codesigning` |
| notarization rejects nested/unsigned code | something sealed code as a resource (usually stray `--deep`) → the script signs inside-out without `--deep`; check you have no extra nested binaries |
| App shows “damaged”/blocked **offline** despite success | the `.app` (or `.dmg`) wasn't stapled → the script staples both; re-run and confirm `stapler validate` passes |
| `resource fork, Finder information, or similar detritus not allowed` | stale xattrs → the script runs `xattr -cr` first; if it persists, clean the source tree |
| See why notarization failed | `xcrun notarytool log <submission-id> --keychain-profile primitive-notary` (the id is printed by `--wait`) |

## Sources

- notarytool migration (altool is EOL): <https://developer.apple.com/documentation/technotes/tn3147-migrating-to-the-latest-notarization-tool>
- `--deep` discouraged for signing: TN2206 <https://developer.apple.com/library/archive/technotes/tn2206/_index.html>
- hardened runtime / omit-if-false entitlements: <https://developer.apple.com/documentation/security/hardened-runtime>
- resolving common notarization issues (timestamp, get-task-allow): <https://developer.apple.com/documentation/security/resolving-common-notarization-issues>
- staple the app AND the dmg: <https://deciphertools.com/blog/notarizing-dmg>
