#!/bin/bash

# Check if a directory was provided, otherwise use current directory
SEARCH_DIR="${1:-.}"

echo "Starting recursive dereference in: $SEARCH_DIR"

find "$SEARCH_DIR" -type l | while read -r link; do
    # Resolve the target
    target=$(readlink -f "$link")

    if [ -e "$target" ]; then
        # Remove symlink and copy the actual data
        rm "$link"
        cp -a "$target" "$link"
        echo "Successfully converted: $link"
    else
        echo "Warning: Skipping broken link: $link"
    fi
done

echo "Done."