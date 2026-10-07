#!/usr/bin/env bash
# Downloads the public datasets used by SG2 (Week 5) into datasets/raw/sg2/
# (datasets/raw/ is git-ignored). Needs: curl, unzip, bsdtar (macOS built-in;
# on Ubuntu/Jetson: sudo apt install libarchive-tools).  ~700 MB in total.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../raw" 2>/dev/null || mkdir -p "$(dirname "$0")/../raw" && cd "$(dirname "$0")/../raw"; pwd)/sg2"
mkdir -p "$ROOT" && cd "$ROOT"

# 1. Eyeblink8 + Talking Face blink annotations (Fogelton & Benesova) - TEST/DEV videos
curl -L -C - -o eyeblink8.zip   https://www.blinkingmatters.com/files/upload/research/eyeblink8.zip
curl -L -C - -o talkingFace.zip https://www.blinkingmatters.com/files/upload/research/talkingFace.zip
unzip -q -o eyeblink8.zip -d eyeblink8
unzip -q -o talkingFace.zip -d talkingFace

# 2. MRL Eye Dataset 2018 (Fusek, VSB Ostrava) - CNN training
curl -L -C - -o mrlEyes_2018_01.zip http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip
unzip -q -o mrlEyes_2018_01.zip -d mrl

# 3. CEW eye patches 24x24 (Song et al., NUAA) - CNN training
curl -L -o cew_eyes24.rar "https://drive.usercontent.google.com/download?id=1Z5hZZnkN4VycK-mOOzUX1Zuwty8BEePy&export=download&confirm=t"
mkdir -p cew && bsdtar -xf cew_eyes24.rar -C cew

echo "SG2 data ready in $ROOT"
