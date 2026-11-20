PANDOC=pandoc
SRC_DIR=src
BUILD_DIR=build
DEPLOY_DIR=deploy

# Blog index is GENERATED (not in git). It must exist before make expands
# the page lists below, otherwise a fresh checkout (e.g. CI) misses blog.html.
# NOTE: no pattern rule generates blog.md here — each flavor generates it in
# its own recipe (build: with hidden; public: without), so the file always
# matches the flavor being built.
$(SRC_DIR)/blog.md: FORCE
	@

.PHONY: FORCE
FORCE:

# blog.html is built via explicit rules below (its .md source is generated in
# the flavor recipes), not via the wildcard pattern, since blog.md does not
# exist at parse time on a fresh checkout.
ALL_MD_FILES=$(shell find $(SRC_DIR) -name '*.md' ! -name 'blog.md')
PUBLIC_MD_FILES=$(shell find $(SRC_DIR) -name '*.md' ! -name 'blog.md' -exec sh -c 'head -20 "$$1" | grep -q "^hidden: *true" || echo "$$1"' _ {} \;)

# Generate HTML file paths
ALL_HTML_FILES=$(patsubst $(SRC_DIR)/%.md,$(BUILD_DIR)/%.html,$(ALL_MD_FILES)) $(BUILD_DIR)/blog.html
PUBLIC_HTML_FILES=$(patsubst $(SRC_DIR)/%.md,$(DEPLOY_DIR)/%.html,$(PUBLIC_MD_FILES)) $(DEPLOY_DIR)/blog.html

# Asset files (excluding markdown and template)
ASSET_FILES=$(shell find $(SRC_DIR) -type f ! -name '*.md' ! -name 'template.html' ! -name 'Makefile' ! -name '.*')
BUILD_ASSETS=$(patsubst $(SRC_DIR)/%,$(BUILD_DIR)/%,$(ASSET_FILES))
DEPLOY_ASSETS=$(patsubst $(SRC_DIR)/%,$(DEPLOY_DIR)/%,$(ASSET_FILES))

.PHONY: all build public serve serve_public clean help FORCE

# Default: build everything including hidden content
all: build

# Build local version with all content (including hidden)
build:
	@./scripts/generate_blog_index.sh $(SRC_DIR) $(SRC_DIR)/blog.md true
	@$(MAKE) $(ALL_HTML_FILES) $(BUILD_ASSETS) $(BUILD_DIR)/blog.html

# Build only public content for deployment.
# The public flavor regenerates the blog index WITHOUT hidden posts first,
# then builds all public pages including the (public) blog index page.
public:
	@./scripts/generate_blog_index.sh $(SRC_DIR) $(SRC_DIR)/blog.md false
	@$(MAKE) $(DEPLOY_DIR)/blog.html
	@$(MAKE) $(PUBLIC_HTML_FILES) $(DEPLOY_ASSETS)

# Explicit rules for the generated blog index page (its .md source is
# generated first via the blog.md rule above)
$(BUILD_DIR)/blog.html: $(SRC_DIR)/blog.md $(SRC_DIR)/template.html
	@mkdir -p $(dir $@)
	$(PANDOC) $< \
	  --template=$(SRC_DIR)/template.html \
	  --standalone \
	  --math-method=mathml \
	  --wrap=none \
	  -f markdown+fenced_divs+tex_math_single_backslash \
	  -o $@

$(DEPLOY_DIR)/blog.html: $(SRC_DIR)/blog.md $(SRC_DIR)/template.html
	@mkdir -p $(dir $@)
	$(PANDOC) $< \
	  --template=$(SRC_DIR)/template.html \
	  --standalone \
	  --math-method=mathml \
	  --wrap=none \
	  -f markdown+fenced_divs+tex_math_single_backslash \
	  -o $@

# Rule to convert Markdown → HTML (local build with hidden files)
$(BUILD_DIR)/%.html: $(SRC_DIR)/%.md $(SRC_DIR)/template.html
	@mkdir -p $(dir $@)
	$(PANDOC) $< \
	  --template=$(SRC_DIR)/template.html \
	  --standalone \
	  --math-method=mathml \
	  --wrap=none \
	  -f markdown+fenced_divs+tex_math_single_backslash \
	  -o $@

# Rule to convert Markdown → HTML (deploy build - public only)
$(DEPLOY_DIR)/%.html: $(SRC_DIR)/%.md $(SRC_DIR)/template.html
	@mkdir -p $(dir $@)
	$(PANDOC) $< \
	  --template=$(SRC_DIR)/template.html \
	  --standalone \
	  --math-method=mathml \
	  --wrap=none \
	  -f markdown+fenced_divs+tex_math_single_backslash \
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
	find $(BUILD_DIR) $(DEPLOY_DIR) -name '.DS_Store' -delete 2>/dev/null || true
	rm -rf $(BUILD_DIR)/* $(DEPLOY_DIR)/* 2>/dev/null || true

# Show help
help:
	@echo "Available targets:"
	@echo "  make build        - Build local version (includes hidden files)"
	@echo "  make serve        - Build and serve locally with hidden files"
	@echo "  make public       - Build public version (excludes hidden files)"
	@echo "  make serve_public - Build and serve public version"
	@echo "  make clean        - Remove all built files"
