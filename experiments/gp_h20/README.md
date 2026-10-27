# Frozen GP training implementation

The active model is `configs/gp_rule.json`: seed 5, fixed H20, selected on the separate 840-layout training bank across 30 seeds. Selection used training relocation totals, not independent validation. Caserta was held out from UWSAA development.

The source engine used 51,000 fitness calls per seed, rather than the paper's reported 50,000. The source attribution, this budget difference, and the training-only selection boundary remain part of the method's provenance.

This directory retains training source, requirements, the frozen protocol and training bank because the current GP runtime fingerprints these inputs. Per-seed receipts, intermediate fixed-H results, execution logs and training-process delivery files are maintained outside the public repository. They are not required to execute the already frozen model or verify published endpoint arithmetic.

To repeat the complete training-receipt review against an archived copy, run `checks/independent_gp_main_review.py --training-receipts <external-receipt-directory>`. Without this argument, the checker verifies available public provenance and explicitly reports that the 30-seed receipt check was not run. No training is launched by the published-result checkers.
