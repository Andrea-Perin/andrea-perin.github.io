#!/bin/bash

SRC_DIR=$1
OUTPUT_FILE=$2
INCLUDE_HIDDEN=$3

echo "% Blog" > "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "# Posts" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Create temp file for sorting
TEMP_FILE=$(mktemp)

# Find all blog posts
find "$SRC_DIR/blog" -name '*.md' -type f | while read -r file; do
    # Check if hidden
    is_hidden=$(head -20 "$file" | grep -q "^hidden: *true" && echo "true" || echo "false")
    
    # Skip hidden files if INCLUDE_HIDDEN is "false"
    if [ "$INCLUDE_HIDDEN" = "false" ] && [ "$is_hidden" = "true" ]; then
        continue
    fi
    
    # Extract title and date from YAML frontmatter
    title=$(grep "^title:" "$file" | head -1 | sed 's/^title: *//' | sed 's/^"\(.*\)"$/\1/' | sed "s/^'\(.*\)'$/\1/")
    date=$(grep "^date:" "$file" | head -1 | sed 's/^date: *//')
    
    # Get relative path
    rel_path=$(echo "$file" | sed "s|^$SRC_DIR/||" | sed 's/\.md$/.html/')
    
    # Format: date|title|path (for sorting)
	echo "$date|$title|$rel_path|$is_hidden" >> "$TEMP_FILE"
done

# Sort by date (newest first) and generate markdown list
sort -r "$TEMP_FILE" | while IFS='|' read -r date title path is_hidden; do
    if [ "$is_hidden" = "true" ]; then
        echo "- **(HIDDEN)** \`$date\` [$title]($path)" >> "$OUTPUT_FILE"
    else
        echo "- \`$date\` [$title]($path)" >> "$OUTPUT_FILE"
    fi
done

# Clean up
rm "$TEMP_FILE"
