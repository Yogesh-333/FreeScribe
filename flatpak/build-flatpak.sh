#!/bin/bash

# Exit on error
set -e

# Ensure we're in the project root directory
if [ ! -f "client_requirements.txt" ]; then
    echo "Error: This script must be run from the project root directory"
    echo "Please run it as: ./flatpak/build-flatpak.sh"
    exit 1
fi

# Check if flatpak is installed
if ! command -v flatpak &> /dev/null; then
    echo "Flatpak is not installed. Please install it first."
    echo "On most Linux distributions, you can install it with:"
    echo "sudo apt install flatpak  # For Debian/Ubuntu"
    echo "sudo dnf install flatpak  # For Fedora"
    echo "sudo pacman -S flatpak   # For Arch Linux"
    exit 1
fi

# Install required flatpak tools if not present
if ! command -v flatpak-builder &> /dev/null; then
    echo "Installing flatpak-builder..."
    if command -v apt &> /dev/null; then
        sudo apt install flatpak-builder
    elif command -v dnf &> /dev/null; then
        sudo dnf install flatpak-builder
    elif command -v pacman &> /dev/null; then
        sudo pacman -S flatpak-builder
    fi
fi

# Add Flathub repository if not already added
if ! flatpak remotes | grep -q "flathub"; then
    echo "Adding Flathub repository..."
    flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
fi

# Install required runtime and SDK
echo "Installing required runtime and SDK..."
# flatpak install flathub org.freedesktop.Platform//24.08 org.freedesktop.Sdk//24.08
flatpak install flathub org.gnome.Platform//48 org.gnome.Sdk//48

# Build the flatpak
# echo "Building FreeScribe Flatpak..."
# flatpak-builder --force-clean build-dir flatpak/io.github.clinicianfocus.FreeScribe.yaml

# Install the flatpak locally
echo "Installing FreeScribe Flatpak locally..."
flatpak-builder --force-clean --user --install build-dir flatpak/io.github.clinicianfocus.FreeScribe.yaml

echo "Build complete! You can now run FreeScribe using:"
echo "flatpak run io.github.clinicianfocus.FreeScribe" 
