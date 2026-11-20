# website2

Minimal pandoc-based academic site. Markdown in `src/` → HTML in `build/` (local) and `deploy/` (public).

## Make targets

| Target | What it does |
|---|---|
| `make build` | Build **everything** into `build/`, *including* posts marked `hidden: true`. Local preview only — do not deploy from this. |
| `make serve` | `make build`, then serve `build/` at `http://localhost:8000`. Includes hidden content (they're labeled `(HIDDEN)` in the blog index). |
| `make public` | Build **only public** content (no `hidden: true` posts) into `deploy/`. This is what should be deployed. |
| `make serve_public` | `make public`, then serve `deploy/` at `http://localhost:8000`. Preview exactly what the world would see. |
| `make all` | Same as `make build` (default includes hidden content). |
| `make clean` | Delete everything in `build/` and `deploy/`. |

## Hidden posts

Any post with `hidden: true` in its YAML frontmatter is:

- excluded from `make public` (and from the blog index generated for it),
- still built by `make build`/`make serve` so you can preview it locally.

They are only secret as long as they are **not committed to git** — the deploy
workflow builds whatever is in the repo.

## Deploy

Push to `main` → GitHub Action (`.github/workflows/deploy.yml`) builds and
publishes to GitHub Pages (`andrea-perin.github.io`).
