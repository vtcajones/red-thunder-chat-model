#!/usr/bin/env python3
"""Build an app manifest, checking the iOS package against the bundled graph."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def direct_https(text):
    url = urlsplit(text)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.fragment:
        raise ValueError("Use direct HTTPS without credentials or a fragment")
    return text

def asset(path, url):
    size = path.stat().st_size
    if not 1 <= size <= 150_000_000:
        raise ValueError("Model must contain 1..150,000,000 bytes")
    return {"url": direct_https(url), "bytes": size, "sha256": sha(path)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ios-package", type=Path, required=True)
    parser.add_argument("--android", type=Path, required=True)
    parser.add_argument("--ios-url", required=True)
    parser.add_argument("--android-url", required=True)
    parser.add_argument("--revision", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.revision <= 1_000_000:
        parser.error("revision must be in 1..1,000,000")
    value = json.loads((Path(__file__).resolve().parents[1] / "manifests/red-thunder-v1.json").read_text())
    package = args.ios_package
    if sha(package / "Manifest.json") != value["coreML"]["packageManifestSHA256"] or sha(package / "Data/com.apple.CoreML/model.mlmodel") != value["coreML"]["specificationSHA256"]:
        parser.error("Core ML graph/package differs from the app's bundled layout; this is not a weights-only override")
    value["revision"] = args.revision
    value["ios"] = asset(package / "Data/com.apple.CoreML/weights/weight.bin", args.ios_url)
    value["android"] = asset(args.android, args.android_url)
    args.output.write_text(json.dumps(value, indent=2) + "\n")

if __name__ == "__main__":
    main()
