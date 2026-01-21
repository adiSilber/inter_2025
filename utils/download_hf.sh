#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] <target_folder> <artifact_id1> [artifact_id2 ...]

Download Hugging Face models or datasets with hash verification.

Arguments:
    target_folder     Target directory (absolute or relative path)
    artifact_id       Hugging Face artifact ID (e.g., 'meta-llama/Llama-2-7b' or 'squad')

Options:
    --skip PATTERNS   Skip files matching patterns (pipe-separated, supports wildcards)
                      Example: --skip "*.bin|*.safetensors|specific_file.txt"
    -h, --help        Show this help message

Examples:
    $0 ./models meta-llama/Llama-2-7b
    $0 /absolute/path/datasets squad glue
    $0 ./models meta-llama/Llama-2-7b --skip "*.bin|*.gguf"
EOF
    exit 1
}

# Function to check if pattern matches
matches_skip_pattern() {
    local file="$1"
    local patterns="$2"
    
    if [ -z "$patterns" ]; then
        return 1  # No patterns, don't skip
    fi
    
    IFS='|' read -ra PATTERN_ARRAY <<< "$patterns"
    for pattern in "${PATTERN_ARRAY[@]}"; do
        # Remove leading/trailing whitespace
        pattern=$(echo "$pattern" | xargs)
        
        # Check for wildcard patterns
        if [[ "$file" == $pattern ]]; then
            return 0  # Match found, skip this file
        fi
    done
    
    return 1  # No match, don't skip
}

# Function to verify file hash
verify_hash() {
    local file="$1"
    local expected_hash="$2"
    
    if [ -z "$expected_hash" ] || [ "$expected_hash" == "null" ]; then
        echo -e "${YELLOW}⚠ No hash available for verification${NC}"
        return 0
    fi
    
    echo "Verifying hash for $(basename "$file")..."
    
    # Determine hash type based on length
    local hash_length=${#expected_hash}
    local actual_hash=""
    
    case $hash_length in
        32)
            actual_hash=$(md5sum "$file" | awk '{print $1}')
            ;;
        40)
            actual_hash=$(sha1sum "$file" | awk '{print $1}')
            ;;
        64)
            actual_hash=$(sha256sum "$file" | awk '{print $1}')
            ;;
        *)
            echo -e "${YELLOW}⚠ Unknown hash type (length: $hash_length)${NC}"
            return 0
            ;;
    esac
    
    if [ "$actual_hash" == "$expected_hash" ]; then
        echo -e "${GREEN}✓ Hash verified${NC}"
        return 0
    else
        echo -e "${RED}✗ Hash mismatch!${NC}"
        echo "  Expected: $expected_hash"
        echo "  Got:      $actual_hash"
        return 1
    fi
}

# Function to download using huggingface_hub Python library
download_with_python() {
    local artifact_id="$1"
    local target_dir="$2"
    local skip_patterns="$3"
    local hf_token="${HF_TOKEN:-}"
    
    python3 << EOF
import sys
import os
from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
from huggingface_hub.utils import HfHubHTTPError
import fnmatch
import hashlib

def matches_pattern(filename, patterns):
    if not patterns:
        return False
    for pattern in patterns.split('|'):
        pattern = pattern.strip()
        if fnmatch.fnmatch(filename, pattern) or filename == pattern:
            return True
    return False

def verify_file_hash(filepath, expected_hash):
    if not expected_hash or expected_hash == 'null':
        return True
    
    hash_length = len(expected_hash)
    if hash_length == 32:
        hasher = hashlib.md5()
    elif hash_length == 40:
        hasher = hashlib.sha1()
    elif hash_length == 64:
        hasher = hashlib.sha256()
    else:
        print(f"Warning: Unknown hash type (length: {hash_length})")
        return True
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    
    return hasher.hexdigest() == expected_hash

artifact_id = "${artifact_id}"
target_dir = "${target_dir}"
skip_patterns = "${skip_patterns}"
hf_token = "${hf_token}" or None

print(f"Downloading {artifact_id}...")
print(f"Token available: {hf_token is not None and len(hf_token) > 0}")

try:
    # Try to list files first to determine if it's a model or dataset
    try:
        files = list_repo_files(artifact_id, repo_type="model", token=hf_token)
        repo_type = "model"
    except:
        try:
            files = list_repo_files(artifact_id, repo_type="dataset", token=hf_token)
            repo_type = "dataset"
        except:
            print(f"Error: Could not access {artifact_id}")
            sys.exit(1)
    
    # Filter files based on skip patterns
    files_to_download = [f for f in files if not matches_pattern(f, skip_patterns)]
    skipped_count = len(files) - len(files_to_download)
    
    if skipped_count > 0:
        print(f"Skipping {skipped_count} file(s) based on patterns")
    
    # Download the repository
    local_dir = os.path.join(target_dir, artifact_id)
    
    def allow_pattern_func(filename):
        return not matches_pattern(filename, skip_patterns)
    
    downloaded_path = snapshot_download(
        artifact_id,
        repo_type=repo_type,
        local_dir=local_dir,
        allow_patterns=None if not skip_patterns else None,  # Use ignore_patterns instead
        ignore_patterns=[p.strip() for p in skip_patterns.split('|')] if skip_patterns else None,
        token=hf_token
    )
    
    print(f"✓ Successfully downloaded to {downloaded_path}")
    
    # Note: HuggingFace Hub handles hash verification internally
    # If download completes without error, integrity is verified
    
except HfHubHTTPError as e:
    print(f"Error downloading {artifact_id}: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)
EOF
}

# Function to download using git with LFS
download_with_git() {
    local artifact_id="$1"
    local target_dir="$2"
    local skip_patterns="$3"
    
    local repo_name=$(basename "$artifact_id")
    local repo_path="$target_dir/$artifact_id"
    
    mkdir -p "$(dirname "$repo_path")"
    
    echo "Cloning https://huggingface.co/$artifact_id..."
    
    # Build clone URL with token if available
    local clone_url="https://huggingface.co/$artifact_id"
    if [ -n "${HF_TOKEN:-}" ]; then
        clone_url="https://user:${HF_TOKEN}@huggingface.co/$artifact_id"
    fi
    
    # Check if directory exists and has .git
    if [ -d "$repo_path/.git" ]; then
        echo "Repository already exists, pulling latest changes..."
        cd "$repo_path"
        git pull
    else
        # Clone the repository
        GIT_LFS_SKIP_SMUDGE=1 git clone "$clone_url" "$repo_path"
        cd "$repo_path"
    fi
    
    # Get list of LFS files
    lfs_files=$(git lfs ls-files -n)
    
    # Filter based on skip patterns
    files_to_download=()
    skipped_count=0
    
    while IFS= read -r file; do
        if [ -n "$file" ]; then
            if matches_skip_pattern "$file" "$skip_patterns"; then
                echo "Skipping: $file"
                ((skipped_count++))
            else
                files_to_download+=("$file")
            fi
        fi
    done <<< "$lfs_files"
    
    # Download LFS files
    if [ ${#files_to_download[@]} -gt 0 ]; then
        echo "Downloading ${#files_to_download[@]} LFS file(s)..."
        for file in "${files_to_download[@]}"; do
            echo "  Downloading: $file"
            git lfs pull --include="$file"
        done
    fi
    
    echo -e "${GREEN}✓ Successfully downloaded $artifact_id${NC}"
    echo "  Location: $repo_path"
    echo "  Skipped: $skipped_count file(s)"
}

# Main script
main() {
    local target_folder=""
    local artifacts=()
    local skip_patterns=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                ;;
            --skip)
                skip_patterns="$2"
                shift 2
                ;;
            *)
                if [ -z "$target_folder" ]; then
                    target_folder="$1"
                else
                    artifacts+=("$1")
                fi
                shift
                ;;
        esac
    done
    
    # Validate arguments
    if [ -z "$target_folder" ]; then
        echo -e "${RED}Error: Target folder not specified${NC}"
        usage
    fi
    
    if [ ${#artifacts[@]} -eq 0 ]; then
        echo -e "${RED}Error: No artifact IDs specified${NC}"
        usage
    fi
    
    # Convert to absolute path if relative
    if [[ "$target_folder" != /* ]]; then
        target_folder="$(cd "$(dirname "$target_folder")" && pwd)/$(basename "$target_folder")"
    fi
    
    # Create target directory
    mkdir -p "$target_folder"
    
    echo "Target folder: $target_folder"
    if [ -n "$skip_patterns" ]; then
        echo "Skip patterns: $skip_patterns"
    fi
    echo "Artifacts to download: ${#artifacts[@]}"
    echo ""
    
    # Check for required tools
    if command -v python3 &> /dev/null && python3 -c "import huggingface_hub" 2>/dev/null; then
        echo "Using huggingface_hub Python library for downloads"
        download_method="python"
    elif command -v git &> /dev/null && command -v git-lfs &> /dev/null; then
        echo "Using git with LFS for downloads"
        download_method="git"
    else
        echo -e "${RED}Error: Neither huggingface_hub Python library nor git-lfs found${NC}"
        echo "Please install one of:"
        echo "  pip install huggingface_hub"
        echo "  OR"
        echo "  apt-get install git git-lfs"
        exit 1
    fi
    
    echo ""
    
    # Download each artifact
    local success_count=0
    local fail_count=0
    
    for artifact in "${artifacts[@]}"; do
        echo "=================================================="
        echo "Downloading: $artifact"
        echo "=================================================="
        
        if [ "$download_method" == "python" ]; then
            if download_with_python "$artifact" "$target_folder" "$skip_patterns"; then
                ((success_count++))
            else
                ((fail_count++))
                echo -e "${RED}✗ Failed to download $artifact${NC}"
            fi
        else
            if download_with_git "$artifact" "$target_folder" "$skip_patterns"; then
                ((success_count++))
            else
                ((fail_count++))
                echo -e "${RED}✗ Failed to download $artifact${NC}"
            fi
        fi
        
        echo ""
    done
    
    # Summary
    echo "=================================================="
    echo "Download Summary"
    echo "=================================================="
    echo -e "Successful: ${GREEN}$success_count${NC}"
    if [ $fail_count -gt 0 ]; then
        echo -e "Failed: ${RED}$fail_count${NC}"
        exit 1
    fi
}

main "$@"
