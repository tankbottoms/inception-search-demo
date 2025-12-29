#!/bin/bash
# Platform Detection Script
# Detects hardware capabilities and outputs the appropriate Docker profile

set -e

# Detect OS
OS_TYPE=$(uname -s)
ARCH_TYPE=$(uname -m)

# Initialize capabilities
HAS_NVIDIA_GPU=false
HAS_APPLE_SILICON=false
PLATFORM="cpu"
PROFILE="cpu"

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_NVIDIA_GPU=true
        PLATFORM="cuda"
        PROFILE="gpu"
    fi
fi

# Check for Apple Silicon
if [[ "$OS_TYPE" == "Darwin" && "$ARCH_TYPE" == "arm64" ]]; then
    HAS_APPLE_SILICON=true
    PLATFORM="apple-silicon"
    PROFILE="cpu"
fi

# Check for Linux ARM64 (DGX Spark without GPU access)
if [[ "$OS_TYPE" == "Linux" && "$ARCH_TYPE" == "aarch64" ]]; then
    if ! $HAS_NVIDIA_GPU; then
        PLATFORM="arm64-cpu"
        PROFILE="cpu"
    fi
fi

# Output based on requested info
case "${1:-profile}" in
    profile)
        echo "$PROFILE"
        ;;
    platform)
        echo "$PLATFORM"
        ;;
    json)
        cat << EOF
{
    "os": "$OS_TYPE",
    "arch": "$ARCH_TYPE",
    "platform": "$PLATFORM",
    "profile": "$PROFILE",
    "hasNvidiaGpu": $HAS_NVIDIA_GPU,
    "hasAppleSilicon": $HAS_APPLE_SILICON
}
EOF
        ;;
    all)
        echo "OS:             $OS_TYPE"
        echo "Architecture:   $ARCH_TYPE"
        echo "Platform:       $PLATFORM"
        echo "Docker Profile: $PROFILE"
        echo "NVIDIA GPU:     $HAS_NVIDIA_GPU"
        echo "Apple Silicon:  $HAS_APPLE_SILICON"
        ;;
    *)
        echo "Usage: $0 [profile|platform|json|all]"
        exit 1
        ;;
esac
