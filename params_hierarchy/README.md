# Pipeline Params Hierarchy

This folder stores the JSON parameter hierarchy metadata for SMF top-level pipelines.

Maintenance rule:

- Whenever the main SMF package adds a new Pipeline, add its `*_pipeline_params_hierarchy.json` here.
- Whenever any Pipeline config parameter is added, removed, renamed, or its dependency/default behavior changes, update the matching JSON here in the same change.
- Keep the top-level JSON structure grouped by `required` and `optional`, then preserve the pipeline-specific category hierarchy underneath.
- These files are intended to support documentation, GUI/schema generation, and human review of Pipeline configuration relationships.
