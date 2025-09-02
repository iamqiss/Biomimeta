#!/usr/bin/env python3
"""
bootstrap_datasets.py
Downloads open video datasets (Blender/Netflix) and prepares short 5s clips
for Biomimeta research scaffolding.
"""

import os
import subprocess
import urllib.request

# Paths
BASE_DIR = "data"
BAV_DIR = os.path.join(BASE_DIR, "bav/stimuli/natural")
CPA_NETFLIX_DIR = os.path.join(BASE_DIR, "cpa/netflix")
PQS_DIR = os.path.join(BASE_DIR, "pqs/clips")

# Ensure folders exist
for d in [BAV_DIR, CPA_NETFLIX_DIR, PQS_DIR]:
    os.makedirs(d, exist_ok=True)

# Sources (small, open test clips)
SOURCES = {
    "bunny": {
        "url": "http://distribution.bbb3d.renderfarming.net/video/mp4/bbb_sunflower_1080p_30fps_normal.mp4",
        "out": os.path.join(BAV_DIR, "big_buck_bunny.mp4"),
    },
    "tears": {
        "url": "https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4",
        "out": os.path.join(CPA_NETFLIX_DIR, "sintel_trailer.mp4"),
    },
    "tos": {
        "url": "https://download.blender.org/durian/movies/ToS-4k-1920.mov",
        "out": os.path.join(PQS_DIR, "tears_of_steel.mov"),
    },
}

def download_file(url, dest):
    if os.path.exists(dest):
        print(f"✔ Already downloaded: {dest}")
        return
    print(f"⬇ Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)

def trim_video(input_path, output_path, duration=5):
    """Trim first N seconds using ffmpeg."""
    print(f"✂ Trimming {input_path} -> {output_path}")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ss", "0", "-t", str(duration),
        "-c:v", "libx264", "-c:a", "aac",
        output_path
    ]
    subprocess.run(cmd, check=True)

def main():
    # Download all sources
    for key, src in SOURCES.items():
        download_file(src["url"], src["out"])

    # Trim into test clips
    trim_video(SOURCES["bunny"]["out"], os.path.join(BAV_DIR, "clip01.mp4"))
    trim_video(SOURCES["tears"]["out"], os.path.join(CPA_NETFLIX_DIR, "clip001.mp4"))
    trim_video(SOURCES["tos"]["out"], os.path.join(PQS_DIR, "clip01_10s.mp4"))

    print("✅ Dataset bootstrap complete!")

if __name__ == "__main__":
    main()
