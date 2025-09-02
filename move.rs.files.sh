#!/bin/bash

# Script to move all .rs files from afiyah subdirectories to the root directory

# Define the source directory
SRC_DIR="afiyah"

# Check if the afiyah directory exists
if [ ! -d "$SRC_DIR" ]; then
    echo "Error: Directory '$SRC_DIR' does not exist."
        exit 1
        fi

        # Find all .rs files in afiyah and its subdirectories
        find "$SRC_DIR" -type f -name "*.rs" | while read -r file; do
            # Get the base filename
                filename=$(basename "$file")
                    
                        # Define the destination path (root directory)
                            dest="./$filename"
                                
                                    # Check if a file with the same name already exists in the root
                                        if [ -f "$dest" ]; then
                                                echo "Warning: File '$dest' already exists. Skipping '$file'."
                                                    else
                                                            # Move the file to the root directory
                                                                    mv "$file" "$dest"
                                                                            echo "Moved: $file -> $dest"
                                                                                fi
                                                                                done

                                                                                # Optionally, remove empty directories in afiyah (uncomment if desired)
                                                                                # find "$SRC_DIR" -type d -empty -delete

                                                                                echo "All .rs files have been moved to the root directory."