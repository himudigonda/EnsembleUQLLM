#!/bin/bash

# Function to recursively list and display file content
function dump_files() {
    local dir="$1"

    # List all files and directories in the current directory
    for file in "$dir"/*; do
        if [ -d "$file" ]; then
            # If it's a directory, recursively call the function
            echo -e "\n--- Entering directory: $file ---"
            dump_files "$file"
        elif [ -f "$file" ]; then
            # If it's a file, print its name and content
            echo -e "\n--- Start of $file ---"
            cat "$file"
            echo -e "\n--- End of $file ---"
        fi
    done
}

# Start from the current directory or a given directory
start_dir="${1:-.}"
dump_files "$start_dir"
