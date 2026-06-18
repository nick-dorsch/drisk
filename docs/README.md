# drisk docs

This folder is the Quarto website root for `drisk` examples and report outputs.

## Rendering the site locally

Quarto is distributed as a CLI tool rather than a normal Python package dependency. Install the Quarto CLI from <https://quarto.org/docs/get-started/> or through your system/tool manager, then render the site from the repository root:

```bash
uv sync --group docs
uv run quarto render docs
```

To preview locally while editing:

```bash
uv run quarto preview docs
```

The rendered website is written to `docs/_site/`.

## Deployment

The site is deployed to GitHub Pages at <https://nick-dorsch.github.io/drisk/> by the workflow in `.github/workflows/docs.yml` whenever changes are pushed to `main`. In GitHub, configure **Settings → Pages → Build and deployment → Source** to **GitHub Actions**.
