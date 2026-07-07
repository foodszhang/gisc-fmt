# SSQ-FMT Evidence Audit

- generator_repo_head_at_generation: `92496da`
- generator_script_git_blob_sha: `666e59823bdb47aa70f0c886145e83dc2bb641fe`
- generator_script_sha256: `ff22009048099b6f805fa3328200a35ffd1173713b5f88bf89678cc5676e6657`
- generation_command: `uv run python tools/visualization/ssq_paper/generate_ssq_paper_figures.py`

## fig_source_separation_profiles

- intended claim: SSQ-FMT separates adjacent and weak multi-source cases better than ablations.
- exact data used: outputs/paper_figures/ssq/qualitative_cases/source_separation_profile_metrics.csv
- sample-selection rule: Hard component success constraints for selected adjacent and three-source cases.
- population/subset: Fixed test300 multi-source samples satisfying selection constraints.
- metric definition: Oblique-plane profiles, valley ratio, bridge area, weak-focus retention, component outcome.
- effect size: See source_separation_profile_metrics.csv.
- confidence interval: Not applicable to case study.
- statistical test: Not applicable to case study.
- supports claim: case-level visual evidence only
- placement: main text candidate

## fig_module_rescue_transitions

- intended claim: SSQ-FMT rescues complete multi-source recovery failures made by ablations.
- exact data used: outputs/paper_figures/ssq/paired_improvement/module_rescue_transitions.csv
- sample-selection rule: GT component count 2/3 and minimum source distance <= 8 mm.
- population/subset: 100 hard multi-source test samples.
- metric definition: Complete recovery success event from component evaluator.
- effect size: [{'label': 'No footprint', 'net_rescued': 9}, {'label': 'No reliability', 'net_rescued': 6}, {'label': 'No source cue', 'net_rescued': 3}]
- confidence interval: [{'label': 'No footprint', 'net_rescued_ci95_low': -1.0, 'net_rescued_ci95_high': 19.0}, {'label': 'No reliability', 'net_rescued_ci95_low': -4.0, 'net_rescued_ci95_high': 16.0}, {'label': 'No source cue', 'net_rescued_ci95_low': -6.0, 'net_rescued_ci95_high': 12.0}]
- statistical test: [{'label': 'No footprint', 'mcnemar_exact_p': 0.10775214433670044}, {'label': 'No reliability', 'mcnemar_exact_p': 0.30745625495910645}, {'label': 'No source cue', 'mcnemar_exact_p': 0.6636238098144531}]
- supports claim: not consistently supported
- placement: supplementary or unsupported

## fig_ssq_module_mechanisms

- intended claim: Mechanistic evidence for PTFA, query-canonical reliability, and source cue.
- exact data used: not generated
- sample-selection rule: requires analysis-mode internal exports
- population/subset: not evaluated
- metric definition: actual footprint, reliability, and source-cue internals
- effect size: not available
- confidence interval: not available
- statistical test: not available
- supports claim: unsupported: actual internal exports are unavailable
- placement: rejected

## fig_module_robustness

- intended claim: Inference-time robustness to detector and view perturbations.
- exact data used: not generated
- sample-selection rule: requires controlled perturbation inference outputs
- population/subset: not evaluated
- metric definition: Dice, complete recovery rate, merge/sample under perturbation
- effect size: not available
- confidence interval: not available
- statistical test: not available
- supports claim: unsupported: perturbation inference has not been run
- placement: rejected
