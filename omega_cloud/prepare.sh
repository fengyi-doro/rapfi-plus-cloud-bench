#!/usr/bin/env bash
set -euo pipefail
mkdir -p dist/{include,licenses}
version=1.22.1
# Microsoft's NuGet distribution includes both the Linux CPU library and its matching C++ headers.
curl -fsSL --retry 4 -o "$RUNNER_TEMP/ort.zip" "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$version/microsoft.ml.onnxruntime.$version.nupkg"
unzip -jo "$RUNNER_TEMP/ort.zip" 'build/native/include/*' -d dist/include
unzip -jo "$RUNNER_TEMP/ort.zip" 'runtimes/linux-x64/native/libonnxruntime.so' -d dist
unzip -jo "$RUNNER_TEMP/ort.zip" 'LICENSE*' 'ThirdPartyNotices*' -d dist/licenses || true
ln -sf libonnxruntime.so dist/libonnxruntime.so.1
curl -fsSL --retry 4 -o "$RUNNER_TEMP/KATAGOMO26.zip" https://download.gomocup.com/ai/KATAGOMO26.zip
echo "48b3d68e1907f70200b3f04a0534b52067ce83bc1d16fc48f7e56506eb39c026  $RUNNER_TEMP/KATAGOMO26.zip" | sha256sum -c
unzip -jo "$RUNNER_TEMP/KATAGOMO26.zip" standard15x.onnx renju15x.onnx -d dist
(cd dist && sha256sum -c <<'HASHES'
126d2c257d68bf6d9935e8fcdd99247334255040cb1d02373d3ad28afa9376ba  standard15x.onnx
25274c71a97f5087bd45efbe7bd2092fb4d66655e5cf7750ee99e71af0adcc15  renju15x.onnx
HASHES
)
sha256sum "$RUNNER_TEMP/ort.zip" dist/libonnxruntime.so
