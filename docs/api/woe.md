# Modeling_Tool.WOE

WOE encoding layer: the `WOE_Master` controller, the `MonotoneWOEBinner`, transformers, plotters, and the adapter that lets
feature screening reuse either binning engine. User guides: [WOE Encoding](../guides/woe.md) and
[WOE Binning Engine](../guides/woe_binning_engine.md).

## Master Class: `WOE_Master`

::: Modeling_Tool.WOE.WOE_Master

## Monotone Binner: `WOE_Monotone_Binner`

!!! note "Large module"

    `MonotoneWOEBinner` is by far the largest class in SMF, so its section is long. It supports chi-square initial binning
    (`fit(chi2_binning=True)`), decision-tree and chi-square refinement (`refine_dtree`, `refine_chi2`), categorical
    clustering (`refine_cate`), special-value governance, and process-pool parallelism (`n_jobs`).

::: Modeling_Tool.WOE.WOE_Monotone_Binner

## Engine Adapter: `WOE_Adapter`

`as_woe_engine` wraps a fitted `WOE_Master` or `MonotoneWOEBinner` behind one interface for the screening tools.

::: Modeling_Tool.WOE.WOE_Adapter

## Transformers and Monotonicity: `WOE_Tool`

::: Modeling_Tool.WOE.WOE_Tool

## Plotting: `WOE_Plot_Tool`

::: Modeling_Tool.WOE.WOE_Plot_Tool

## WOE Excel Report: `WOE_Report_Builder`

::: Modeling_Tool.WOE.WOE_Report_Builder

## Group Value Helpers: `plot_woe_tool`

::: Modeling_Tool.WOE.plot_woe_tool
