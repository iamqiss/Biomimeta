#!/bin/bash

# Script to insert COPYRIGHT.txt content into the top of every .rs file in top-level directories

# Define the copyright file
COPYRIGHT_FILE="COPYRIGHT.txt"

# Check if COPYRIGHT.txt exists
if [ ! -f "$COPYRIGHT_FILE" ]; then
    echo "Error: $COPYRIGHT_FILE does not exist in the current directory."
        exit 1
        fi

        # Read the content of COPYRIGHT.txt
        COPYRIGHT_CONTENT=$(cat "$COPYRIGHT_FILE")

        # List of top-level directories (previously moved from afiyah)
        TOP_LEVEL_DIRS=(
            "retinal_processing"
                "cortical_processing"
                    "synaptic_adaptation"
                        "perceptual_optimization"
                            "multi_modal_integration"
                                "experimental_features"
                                    "streaming_engine"
                                        "utilities"
                                            "configs"
                                            )

                                            # Flag to track if any files were modified
                                            MODIFIED=0

                                            # Iterate through each top-level directory
                                            for dir in "${TOP_LEVEL_DIRS[@]}"; do
                                                # Check if the directory exists
                                                    if [ -d "$dir" ]; then
                                                            # Find all .rs files in the directory and its subdirectories
                                                                    find "$dir" -type f -name "*.rs" | while read -r file; do
                                                                                # Check if the file already contains the copyright notice
                                                                                            if grep -Fx "$COPYRIGHT_CONTENT" "$file" > /dev/null; then
                                                                                                            echo "Skipping: $file (already contains copyright)"
                                                                                                                        else
                                                                                                                                        # Create a temporary file
                                                                                                                                                        TEMP_FILE=$(mktemp)
                                                                                                                                                                        
                                                                                                                                                                                        # Prepend copyright content to the file
                                                                                                                                                                                                        echo -e "$COPYRIGHT_CONTENT\n$(cat "$file")" > "$TEMP_FILE"
                                                                                                                                                                                                                        
                                                                                                                                                                                                                                        # Move the temporary file back to the original
                                                                                                                                                                                                                                                        mv "$TEMP_FILE" "$file"
                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                        echo "Updated: $file"
                                                                                                                                                                                                                                                                                                        MODIFIED=1
                                                                                                                                                                                                                                                                                                                    fi
                                                                                                                                                                                                                                                                                                                            done
                                                                                                                                                                                                                                                                                                                                else
                                                                                                                                                                                                                                                                                                                                        echo "Warning: Directory '$dir' does not exist."
                                                                                                                                                                                                                                                                                                                                            fi
                                                                                                                                                                                                                                                                                                                                            done

                                                                                                                                                                                                                                                                                                                                            # Check if any files were modified
                                                                                                                                                                                                                                                                                                                                            if [ $MODIFIED -eq 0 ]; then
                                                                                                                                                                                                                                                                                                                                                echo "No files were modified (all .rs files already contain the copyright or no .rs files found)."
                                                                                                                                                                                                                                                                                                                                                else
                                                                                                                                                                                                                                                                                                                                                    echo "Copyright added to all applicable .rs files."
                                                                                                                                                                                                                                                                                                                                                    fi