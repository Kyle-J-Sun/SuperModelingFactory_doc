# ExcelMaster

Excel report engine. `ExcelMaster` wraps [xlsxwriter](https://xlsxwriter.readthedocs.io/) and adds a cursor: every write
advances the current row, so you stack tables, images, and charts on a sheet without computing cell coordinates. It also
provides charts and conditional formatting. User guide: [Excel Report Generation](../guides/excel_report.md).

The package has no top-level exports. Import the class from its module:

```python
from ExcelMaster.ExcelMaster import ExcelMaster
```

## Cell Formats: `ExcelFormatTool`

::: ExcelMaster.ExcelFormatTool

## Writer: `ExcelMaster`

::: ExcelMaster.ExcelMaster

## Report Templates: `Template`

::: ExcelMaster.Template

## Utilities: `Utility`

::: ExcelMaster.Utility
