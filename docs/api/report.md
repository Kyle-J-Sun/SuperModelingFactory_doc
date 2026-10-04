# Report

Report templates. `Report_Tool` takes the artifacts a modeling run leaves on disk (performance CSVs, performance and WOE
plot images, variable-importance tables) and lays them out as formatted sheets with [ExcelMaster](excelmaster.md). It does no
modeling or data transformation itself. User guide: [Excel Report Generation](../guides/excel_report.md).

Every function takes an `ExcelMaster` instance and a worksheet first and writes at the cursor. Import the functions from
their module:

```python
from Report.Report_Tool import single_model_perf, get_model_varimp
```

## Report Functions: `Report_Tool`

::: Report.Report_Tool
