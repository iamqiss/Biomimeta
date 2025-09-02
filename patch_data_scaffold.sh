#!/usr/bin/env bash
# patch_data_scaffold.sh
# Extends Biomimeta research repo with data/ folder + dummy assets + manifests

set -e

echo "📂 Creating data/ directories..."
mkdir -p data/bav/stimuli/{gratings,natural,saccades}
mkdir -p data/cpa/{netflix,uvg}
mkdir -p data/pqs/clips

echo "📦 Adding dummy files..."
# BAV
echo "PNG_PLACEHOLDER" > data/bav/stimuli/gratings/grating_low.png
echo '{"saccades":[{"onset":0.1,"amplitude":2.3}]}' > data/bav/stimuli/saccades/saccade1.json
echo "MP4_PLACEHOLDER" > data/bav/stimuli/natural/clip01.mp4

# CPA
echo "MP4_PLACEHOLDER" > data/cpa/netflix/clip001.mp4
echo "MP4_PLACEHOLDER" > data/cpa/uvg/clip001.mp4

# PQS
echo "MP4_PLACEHOLDER" > data/pqs/clips/clip01_10s.mp4
echo "MP4_PLACEHOLDER" > data/pqs/clips/clip02_10s.mp4

echo "📝 Writing manifests with real paths..."

# BAV manifest
cat > studies/bav/stimuli/manifest.yaml <<'EOF'
stimuli:
  - id: grating_low
    path: ../../data/bav/stimuli/gratings/grating_low.png
    type: grating
  - id: natural_clip_01
    path: ../../data/bav/stimuli/natural/clip01.mp4
    type: natural
  - id: saccade_profile_01
    path: ../../data/bav/stimuli/saccades/saccade1.json
    type: saccade
EOF

# CPA manifests
cat > studies/cpa/datasets/manifest_netflix_v1.json <<'EOF'
{
  "name": "netflix_v1",
  "clips": [
    {"id": "clip001", "path": "../../data/cpa/netflix/clip001.mp4", "hash": "dummyhash"}
  ]
}
EOF

cat > studies/cpa/datasets/manifest_uvg_v1.json <<'EOF'
{
  "name": "uvg_v1",
  "clips": [
    {"id": "clip001", "path": "../../data/cpa/uvg/clip001.mp4", "hash": "dummyhash"}
  ]
}
EOF

# PQS manifest
cat > studies/pqs/stimuli_manifests/round1.yaml <<'EOF'
clips:
  - id: clip01
    path: ../../data/pqs/clips/clip01_10s.mp4
  - id: clip02
    path: ../../data/pqs/clips/clip02_10s.mp4
EOF

echo "✅ Data scaffolding complete. You can now swap dummy files with real datasets."
