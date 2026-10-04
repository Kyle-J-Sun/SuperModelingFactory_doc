# Modeling_Tool.UAT

Online/offline consistency checks: `UATConsistencyChecker` compares the online and offline scores and model input features of
the same applications and reports mismatches within configurable tolerances. User guide:
[Online/Offline Consistency Check](../guides/uat.md).

`UATConfig` and `UATConsistencyChecker` are not exported at the top level. Import them from the subpackage:
`from Modeling_Tool.UAT import UATConfig, UATConsistencyChecker`.

## Consistency Checker: `UAT_Consistency_Checker`

::: Modeling_Tool.UAT.UAT_Consistency_Checker
