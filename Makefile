PANDOC=pandoc
SRC_DIR=src
BUILD_DIR=build
DEPLOY_DIR=deploy

# Find all markdown files
ALL_MD_FILES=$(shell find $(SRC_DIR) -name '*.md')

# Find files that DON'T have 'hidden: true' in frontmatter (public files)
PUBLIC_MD_FILES=$(shell find $(SRC_DIR) -name '*.md' -exec sh -c 'head -20 "$$1" | grep -q "^hidden: *true" || echo "$$1"' _ {} \;)

# Generate HTML file paths
ALL_HTML_FILES=$(patsubst $(SRC_DIR)/%.md,$(BUILD_DIR)/%.html,$(ALL_MD_FILES))
PUBLIC_HTML_FILES=$(patsubst $(SRC_DIR)/%.md,$(DEPLOY_DIR)/%.html,$(PUBLIC_MD_FILES))

# Asset files (excluding markdown and template)
ASSET_FILES=$(shell find $(SRC_DIR) -type f ! -name '*.md' ! -name 'template.html' ! -name 'Makefile' ! -name '.*')
BUILD_ASSETS=$(patsubst $(SRC_DIR)/%,$(BUILD_DIR)/%,$(ASSET_FILES))
DEPLOY_ASSETS=$(patsubst $(SRC_DIR)/%,$(DEPLOY_DIR)/%,$(ASSET_FILES))

.PHONY: all build public serve serve_public clean help

# Default: build everything including hidden content
all: build

# Build local version with all content (including hidden)
build: generate_blog_index_local $(ALL_HTML_FILES) $(BUILD_ASSETS)

# Build only public content for deployment
public: generate_blog_index_public $(PUBLIC_HTML_FILES) $(DEPLOY_ASSETS)

# Generate blog index with hidden posts
generate_blog_index_local:
	@./scripts/generate_blog_index.sh $(SRC_DIR) $(SRC_DIR)/blog.md true

# Generate blog index without hidden posts
generate_blog_index_public:
	@./scripts/generate_blog_index.sh $(SRC_DIR) $(SRC_DIR)/blog.md false

# Rule to convert Markdown → HTML (local build with hidden files)
$(BUILD_DIR)/%.html: $(SRC_DIR)/%.md $(SRC_DIR)/template.html $(SRC_DIR)/style.css
	@mkdir -p $(dir $@)
	$(PANDOC) $< \
	  --template=$(SRC_DIR)/template.html \
	  --standalone \
	  --mathjax \
	  -f markdown+fenced_divs \
	  -o $@

# Rule to convert Markdown → HTML (deploy build - public only)
$(DEPLOY_DIR)/%.html: $(SRC_DIR)/%.md $(SRC_DIR)/template.html $(SRC_DIR)/style.css
	@mkdir -p $(dir $@)
	$(PANDOC) $< \
	  --template=$(SRC_DIR)/template.html \
	  --standalone \
	  --mathjax \
	  -f markdown+fenced_divs \
	  -o $@

# Copy assets to build directory
$(BUILD_DIR)/%: $(SRC_DIR)/%
	@mkdir -p $(dir $@)
	@cp $< $@

# Copy assets to deploy directory
$(DEPLOY_DIR)/%: $(SRC_DIR)/%
	@mkdir -p $(dir $@)
	@cp $< $@

# Serve local build with hidden files at port 8000
serve: build
	@echo "Starting local server at http://localhost:8000"
	@echo "This version includes hidden files for preview"
	@echo "Press Ctrl+C to stop"
	@cd $(BUILD_DIR) && python3 -m http.server 8000

# Serve public build (no hidden files) at port 8000
serve_public: public
	@echo "Starting local server at http://localhost:8000"
	@echo "This version includes only public files"
	@echo "Press Ctrl+C to stop"
	@cd $(DEPLOY_DIR) && python3 -m http.server 8000

# Clean build directories
clean:
	rm -rf $(BUILD_DIR)/* $(DEPLOY_DIR)/*

# Show help
help:
	@echo "Available targets:"
	@echo "  make build        - Build local version (includes hidden files)"
	@echo "  make serve        - Build and serve locally with hidden files"
	@echo "  make public       - Build public version (excludes hidden files)"
	@echo "  make serve_public - Build and serve public version"
	@echo "  make clean        - Remove all built files"
