#!/bin/bash

# bio.io Installation Script
# This script helps you set up the bio.io plugin repository

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Print welcome message
echo "================================================"
echo "          bio.io Installation Script           "
echo "================================================"
echo ""

# Check prerequisites
print_info "Checking prerequisites..."

MISSING_DEPS=()

if ! command_exists git; then
    MISSING_DEPS+=("git")
fi

if ! command_exists python3; then
    MISSING_DEPS+=("python3")
fi

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    print_error "Missing required dependencies: ${MISSING_DEPS[*]}"
    echo ""
    echo "Please install the following before running this script:"
    for dep in "${MISSING_DEPS[@]}"; do
        echo "  - $dep"
    done
    exit 1
fi

print_success "All prerequisites are installed"
echo ""

# Determine installation directory
INSTALL_DIR="${INSTALL_DIR:-$HOME/.bio.io}"

echo "Installation Options:"
echo "  1) Install to default location ($INSTALL_DIR)"
echo "  2) Install to current directory ($(pwd))"
echo "  3) Specify custom location"
echo ""

read -p "Select option [1-3] (default: 1): " INSTALL_OPTION
INSTALL_OPTION=${INSTALL_OPTION:-1}

case $INSTALL_OPTION in
    1)
        TARGET_DIR="$INSTALL_DIR"
        ;;
    2)
        TARGET_DIR="$(pwd)/bio.io"
        ;;
    3)
        read -p "Enter installation path: " CUSTOM_PATH
        TARGET_DIR="$CUSTOM_PATH"
        ;;
    *)
        print_error "Invalid option. Exiting."
        exit 1
        ;;
esac

echo ""
print_info "Installing bio.io to: $TARGET_DIR"
echo ""

# Clone or update repository
if [ -d "$TARGET_DIR" ]; then
    print_info "Directory already exists. Updating..."
    cd "$TARGET_DIR"
    
    if [ -d ".git" ]; then
        git pull origin main || git pull origin master || {
            print_error "Failed to update repository"
            exit 1
        }
        print_success "Repository updated"
    else
        print_error "Directory exists but is not a git repository"
        exit 1
    fi
else
    print_info "Cloning repository..."
    git clone https://github.com/princess38827/bio.io.git "$TARGET_DIR" || {
        print_error "Failed to clone repository"
        exit 1
    }
    print_success "Repository cloned"
    cd "$TARGET_DIR"
fi

echo ""

# Count plugins
PLUGIN_COUNT=$(find plugins -maxdepth 1 -type d | wc -l)
PLUGIN_COUNT=$((PLUGIN_COUNT - 1))  # Subtract 1 for the plugins directory itself

print_success "Installation complete!"
echo ""
echo "================================================"
echo "              Setup Summary                     "
echo "================================================"
echo "Installation directory: $TARGET_DIR"
echo "Available plugins: $PLUGIN_COUNT"
echo ""
echo "Next steps:"
echo "  1. Explore available plugins:"
echo "     cd $TARGET_DIR/plugins"
echo "     ls"
echo ""
echo "  2. Read plugin documentation:"
echo "     Each plugin contains SKILL.md files with usage instructions"
echo ""
echo "  3. Add bio.io to your PATH (optional):"
echo "     export PATH=\"$TARGET_DIR:\$PATH\""
echo ""
echo "For more information, visit:"
echo "  https://github.com/princess38827/bio.io"
echo ""
print_success "Happy coding!"
echo ""
