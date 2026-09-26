#!/bin/sh
set -e

WEIGHTS="/root/.deepface/weights/vgg_face_weights.h5"
mkdir -p /root/.deepface/weights

if [ ! -f "$WEIGHTS" ]; then
  echo "Downloading VGG-Face weights (~145 MB)..."
  wget -qO "$WEIGHTS" \
    https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5
  echo "VGG-Face weights ready."
fi

exec "$@"
