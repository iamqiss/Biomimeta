#!/bin/bash

# Script to move all top-level directories from afiyah to the root directory

# Define the source directory
SRC_DIR="afiyah"

# Check if the afiyah directory exists
if [ ! -d "$SRC_DIR" ]; then
    echo "Error: Directory '$SRC_DIR' does not exist."
        exit 1
        fi

        # Find all top-level directories in afiyah (not subdirectories)
        for dir in "$SRC_DIR"/*/ ; do
            # Check if there are any directories
                if [ ! -d "$dir" ]; then
                        echo "No directories found in '$SRC_DIR'."
                                exit 0
                                    fi

                                        # Get the directory name
                                            dir_name=$(basename "$dir")
                                                
                                                    # Define the destination path (root directory)
                                                        dest="./$dir_name"
                                                            
                                                                # Check if a directory with the same name already exists in the root
                                                                    if [ -d "$dest" ]; then
                                                                            echo "Warning: Directory '$dest' already exists. Skipping '$dir'."
                                                                                else
                                                                                        # Move the directory to the root
                                                                                                mv "$dir" "$dest"
                                                                                                        echo "Moved: $dir -> $dest"
                                                                                                            fi
                                                                                                            done

                                                                                                            echo "All top-level directories have been moved to the root directory."