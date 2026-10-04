# SuperModelingFactory Documentation

Source of the documentation site for [SuperModelingFactory (SMF)](https://github.com/Kyle-J-Sun/SuperModelingFactory), a
Python toolkit for credit-risk scorecard development: sample design, WOE binning, feature screening, model training,
evaluation, explainability, and Excel reporting.

**Live site:** <https://kyle-j-sun.github.io/SuperModelingFactory_doc/>

The site is built with [MkDocs](https://www.mkdocs.org/) and the [Material](https://squidfunk.github.io/mkdocs-material/)
theme. API pages are generated from the package's docstrings by `mkdocstrings`.

## Where to start reading

| If you are... | Read |
|---|---|
| New to SMF | **Home** → **Installation** → **Quickstart** (a runnable 5-minute walkthrough on synthetic data) |
| Planning a project | **Architecture** (module map and dependency rules), **End-to-End Pipelines** |
| Doing day-to-day modeling | **User Guides**: one guide per topic (samples, WOE, feature screening, models, evaluation, explainability, reject inference, UAT, Excel reports, ODPS) |
| Looking up a signature | **API Reference** (generated from source) |
| Upgrading | **ChangeLog** (one page per release, with behavior changes called out) |
| Stuck | **FAQ** |

## Preview locally

The API pages read the SMF source tree, so check the two repositories out **side by side** with these exact folder names:

```
workspace/
├── SuperModelingFactory/        # the package
└── SuperModelingFactory_doc/    # this repository
```

```bash
cd SuperModelingFactory_doc
pip install -r requirements-docs.txt
pip install -r ../SuperModelingFactory/requirements.txt

mkdocs serve            # live preview at http://127.0.0.1:8000
mkdocs build --strict   # static site in ./site/ (what CI publishes; fails on warnings)
```

## Repository layout

```
SuperModelingFactory_doc/
├── mkdocs.yml                 # Site configuration: theme, navigation, mkdocstrings (source path ../SuperModelingFactory)
├── requirements-docs.txt      # Documentation toolchain
├── docs/
│   ├── index.md, installation.md, quickstart.md, architecture.md, faq.md
│   ├── pipeline.md            # Step-by-step manual modeling pipeline
│   ├── pipeline_one_click.md  # The seven one-click Pipelines and their config references
│   ├── guides/                # Task-oriented user guides
│   ├── api/                   # API reference pages (mkdocstrings directives)
│   └── changelog/             # Release notes, one file per version
├── params_hierarchy/          # Pipeline parameter hierarchy metadata (JSON), see its README
├── tools/check_examples.py    # Checks (and optionally runs) the Python examples in the Markdown files
└── .github/workflows/pages.yml  # Builds with `mkdocs build` and deploys to GitHub Pages on push to `main`
```

## Contributing documentation

- **Code must run.** Examples are written against the installed SMF release. Keep keyword names, defaults, and import
  paths exactly as the package exposes them; names that are not re-exported at the top level must be imported from their
  subpackage (for example `from Modeling_Tool.Feature import proc_means_by_grp`).
- **Check before you push.** `tools/check_examples.py` statically checks every Python code block in the Markdown files
  against the installed SMF (imports resolve, keywords exist, required arguments are present, result attributes exist):

  ```bash
  python tools/check_examples.py README.md docs/*.md docs/*/*.md
  ```

  Add `--run` to also execute the blocks of a page in order, in one namespace, and report exceptions. Pages written to run
  on their own (such as `docs/quickstart.md` and `docs/guides/eval.md`) need nothing else; a page that continues
  another one can be run with `--prelude FILE`, a Python file that defines the objects it assumes. Mark a snippet that needs
  credentials or a GUI with `# check: skip` as its first line: it is then checked statically but not executed. A call that
  passes `...` as an argument counts as elided pseudo-code (common in release notes): its keywords are checked, but not
  the missing arguments.

  ```bash
  python tools/check_examples.py --run docs/quickstart.md docs/guides/eval.md
  ```

- **Keep headings stable.** Cross-page links use heading anchors (for example `guides/model.md#sample-weights`); `mkdocs build --strict`
  reports broken ones.
- **Update with the code.** A change that adds or changes a public parameter updates the matching guide, the
  `params_hierarchy/*.json` entry for pipeline config fields, and the changelog page for that release.
- **Release order.** For a code change across the four SMF repositories, push in this order: pytest → doc → agent → main.

## The SMF repositories

| Repository | Purpose |
|---|---|
| [SuperModelingFactory](https://github.com/Kyle-J-Sun/SuperModelingFactory) | Package source |
| [SuperModelingFactory_doc](https://github.com/Kyle-J-Sun/SuperModelingFactory_doc) | This documentation |
| [SuperModelingFactory_pytest](https://github.com/Kyle-J-Sun/SuperModelingFactory_pytest) | Regression test suite |
| [SuperModelingFactory_agent](https://github.com/Kyle-J-Sun/SuperModelingFactory_agent) | AI-assistant skill for SMF |
