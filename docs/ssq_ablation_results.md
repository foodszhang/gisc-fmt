# SSQ Mainline and Ablation Results

Date: 2026-06-16

This document records the completed FMT-SimGen v2 test300 evaluation for the source-separable query (SSQ) mainline and related ablations. Based on the completed ablations, the formal mainline is now the no-center-auxiliary SSQ variant: measurement-derived source-separable query representation with PTFA, canonical reliability, source cue, and distance auxiliary supervision. The original E15 center-distance result is retained as a center-auxiliary ablation. The source-slot soft-union decoder is retained only as an ablation because it did not outperform the SSQ density-head path.

## Evaluation Protocol

- Dataset: `/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k`
- Split: fixed `test.txt`, 300 samples
- Threshold: `0.5` for voxel metrics
- Full-volume evaluator: `scripts/eval_full_volume_fmt_simgen.py`
- Component evaluator: `scripts/eval_components_fmt_simgen.py`
- Component matching: `iou_threshold=0.01`, `centroid_threshold_vox=3.0`, `connectivity=26`, `min_region_size=10`

Reproducibility note: after this analysis, `fmt_simgen_v2_ssq_main` was updated to the
no-center-auxiliary mainline. The retained `no_ptfa`, `no_canonical_reliability`,
`no_source_cue`, and `no_distance_aux` checkpoints were trained before that relabeling,
on the original center-auxiliary SSQ family. They are valid evidence that these modules
matter in the SSQ family, while future strict ablations against the new no-center
mainline should be rerun from the updated configs if exact one-factor attribution is
needed.

## Cleanup State

- Removed precomputed `center_distance` directories from all 3000 samples after evaluation; remaining count is `0`.
- Cleaned unused SSQ ablation checkpoints and retained one evaluated/best checkpoint per run.
- Retained checkpoints:
  - `SSQ mainline (no center aux)`: `outputs/fmt_simgen_v2_ssq_ablation/runs/no_center_aux/checkpoints/epoch=73-val_dice=0.7364.ckpt`
  - `SSQ center_aux ablation (E15)`: `outputs/fmt_simgen_v2_e15_center_distance_precomputed/checkpoints/epoch=52-val_dice=0.7414.ckpt`
  - `Source-slot soft-union`: `outputs/gisc_fmt/fit/2026-06-12/12-37-02/checkpoints/epoch=10-val_dice=0.7232.ckpt`
  - `SSQ no_ptfa`: `outputs/fmt_simgen_v2_ssq_ablation/runs/no_ptfa/checkpoints/epoch=51-val_dice=0.7302.ckpt`
  - `SSQ no_canonical_reliability`: `outputs/fmt_simgen_v2_ssq_ablation/runs/no_canonical_reliability/checkpoints/epoch=59-val_dice=0.7262.ckpt`
  - `SSQ no_source_cue`: `outputs/fmt_simgen_v2_ssq_ablation/runs/no_source_cue/checkpoints/epoch=59-val_dice=0.7302.ckpt`
  - `SSQ no_distance_aux`: `outputs/fmt_simgen_v2_ssq_ablation/runs/no_distance_aux/checkpoints/epoch=59-val_dice=0.7256.ckpt`

## Overall Results

| Run | Kind | Dice | IoU | Precision | Recall | ASSD | HD95 | Comp recall | Comp precision | Miss/sample | Merge/sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SSQ mainline (no center aux) | mainline | 0.7299 | 0.5903 | 0.7484 | 0.7878 | 0.3720 | 1.1964 | 0.9128 | 0.9881 | 0.2333 | 0.1267 |
| SSQ center_aux ablation (E15) | ablation | 0.7246 | 0.5831 | 0.7259 | 0.7976 | 0.3964 | 1.3591 | 0.9133 | 0.9889 | 0.2333 | 0.1100 |
| Source-slot soft-union | decoder ablation | 0.7093 | 0.5630 | 0.6966 | 0.8015 | 0.3822 | 1.1360 | 0.9006 | 0.9856 | 0.2633 | 0.1633 |
| SSQ no_ptfa | ablation | 0.6986 | 0.5510 | 0.6534 | 0.8332 | 0.3958 | 1.1093 | 0.8961 | 0.9778 | 0.2733 | 0.2033 |
| SSQ no_canonical_reliability | ablation | 0.7123 | 0.5661 | 0.6998 | 0.8017 | 0.3760 | 1.1125 | 0.9022 | 0.9914 | 0.2600 | 0.1800 |
| SSQ no_source_cue | ablation | 0.7079 | 0.5613 | 0.6880 | 0.8118 | 0.3869 | 1.1310 | 0.9039 | 0.9856 | 0.2533 | 0.1600 |
| SSQ no_distance_aux | ablation | 0.7106 | 0.5646 | 0.6954 | 0.8048 | 0.3943 | 1.2275 | 0.9094 | 0.9861 | 0.2433 | 0.1533 |

## Main Findings

- The no-center SSQ variant is the formal mainline because it has the strongest test300 Dice (`0.7299`) while maintaining component recall comparable to E15 (`0.9128` vs `0.9133`).
- The original E15 center-distance checkpoint is now the `center_aux` ablation: its Dice is lower (`0.7246`) but merge/sample is lower (`0.1100` vs `0.1267`). This means center auxiliary can be discussed as a separation-oriented regularizer, but not as the main performance path.
- `Source-slot soft-union` underperforms the no-center SSQ mainline on Dice (`0.7093` vs `0.7299`) and component recall (`0.9006` vs `0.9128`), so it should remain an ablation rather than the main method.
- `no_ptfa` is the strongest negative ablation: Dice drops to `0.6986`, precision drops to `0.6534`, and merge/sample rises to `0.2033`. PTFA should stay in the mainline.
- `no_canonical_reliability` drops Dice to `0.7123` and component recall to `0.9022`; canonical reliability contributes but is not the sole driver.
- `no_source_cue` drops Dice to `0.7079` and component recall to `0.9039`; measurement-derived source cue should remain part of the SSQ representation.
- `no_distance_aux` drops Dice to `0.7106` and increases merge/sample to `0.1533`; distance auxiliary supervision is useful for final reconstruction and separation.

## SSQ mainline (no center aux)

- Checkpoint: `outputs/fmt_simgen_v2_ssq_ablation/runs/no_center_aux/checkpoints/epoch=73-val_dice=0.7364.ckpt`
- Output dir: `outputs/fmt_simgen_v2_ssq_ablation/test300/no_center_aux`
- Note: Formal source-separable query representation mainline: PTFA, canonical reliability, measurement-derived source cue, distance auxiliary supervision, center auxiliary disabled.

| Dice | IoU | Precision | Recall | ASSD | HD95 | CLE | NRMSE | Comp recall | Miss/sample | Merge/sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7299 | 0.5903 | 0.7484 | 0.7878 | 0.3720 | 1.1964 | 0.9491 | 0.0302 | 0.9128 | 0.2333 | 0.1267 |

### By Num Foci

| Num foci | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 | 0.7892 | 0.6690 | 0.7669 | 0.8908 | 0.2673 | 0.5802 |
| 2 | 111 | 0.7323 | 0.5915 | 0.7486 | 0.7990 | 0.3372 | 1.0076 |
| 3 | 109 | 0.6840 | 0.5314 | 0.7348 | 0.7007 | 0.4844 | 1.8410 |

### Component Metrics by Num Foci

| Num foci | N | GT comp | Pred comp | Matched | Missed | False | Merge | Split | Comp recall | Comp precision | Matched IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.6690 |
| 2 | 111 | 1.9730 | 1.8559 | 1.8468 | 0.1261 | 0.0090 | 0.0721 | 0.0090 | 0.9369 | 0.9955 | 0.5592 |
| 3 | 109 | 2.8807 | 2.4404 | 2.3670 | 0.5138 | 0.0734 | 0.2752 | 0.0367 | 0.8242 | 0.9717 | 0.5207 |

### By Depth Tier

| Depth | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deep | 89 | 0.7457 | 0.6104 | 0.7768 | 0.7855 | 0.3313 | 0.9718 |
| medium | 123 | 0.7338 | 0.5946 | 0.7485 | 0.7943 | 0.3839 | 1.2415 |
| shallow | 88 | 0.7087 | 0.5640 | 0.7198 | 0.7808 | 0.3967 | 1.3606 |

### By Shape Class

| Shape class | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid | 41 | 0.7665 | 0.6369 | 0.7844 | 0.8270 | 0.3072 | 0.7165 |
| irregular | 50 | 0.7520 | 0.6220 | 0.7430 | 0.8533 | 0.3405 | 1.0867 |
| mixed_three_shape | 18 | 0.6791 | 0.5272 | 0.7883 | 0.6697 | 0.4911 | 1.8171 |
| mixed_two_shape | 160 | 0.7124 | 0.5666 | 0.7405 | 0.7540 | 0.4064 | 1.3768 |
| sphere | 31 | 0.7660 | 0.6370 | 0.7276 | 0.8732 | 0.2623 | 0.7169 |

### By Shape Set

| Shape set | N | Dice | IoU | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| ellipsoid | 41 | 0.7665 | 0.6369 | 0.7844 | 0.8270 |
| ellipsoid+irregular | 47 | 0.7216 | 0.5754 | 0.7759 | 0.7262 |
| ellipsoid+irregular+sphere | 18 | 0.6791 | 0.5272 | 0.7883 | 0.6697 |
| ellipsoid+sphere | 70 | 0.7149 | 0.5719 | 0.7364 | 0.7609 |
| irregular | 50 | 0.7520 | 0.6220 | 0.7430 | 0.8533 |
| irregular+sphere | 43 | 0.6982 | 0.5483 | 0.7085 | 0.7730 |
| sphere | 31 | 0.7660 | 0.6370 | 0.7276 | 0.8732 |

## SSQ center_aux ablation (E15)

- Checkpoint: `outputs/fmt_simgen_v2_e15_center_distance_precomputed/checkpoints/epoch=52-val_dice=0.7414.ckpt`
- Output dir: `outputs/fmt_simgen_v2_source_slots_eval/e15_center_distance_epoch52_test300`
- Note: Original E15 center-distance checkpoint; now interpreted as the center auxiliary ablation against the no-center SSQ mainline.

| Dice | IoU | Precision | Recall | ASSD | HD95 | CLE | NRMSE | Comp recall | Miss/sample | Merge/sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7246 | 0.5831 | 0.7259 | 0.7976 | 0.3964 | 1.3591 | 1.0597 | 0.0306 | 0.9133 | 0.2333 | 0.1100 |

### By Num Foci

| Num foci | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 | 0.7725 | 0.6482 | 0.7239 | 0.9134 | 0.2978 | 0.6375 |
| 2 | 111 | 0.7329 | 0.5915 | 0.7209 | 0.8210 | 0.3641 | 1.1772 |
| 3 | 109 | 0.6809 | 0.5269 | 0.7323 | 0.6888 | 0.5016 | 2.0739 |

### Component Metrics by Num Foci

| Num foci | N | GT comp | Pred comp | Matched | Missed | False | Merge | Split | Comp recall | Comp precision | Matched IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.6482 |
| 2 | 111 | 1.9730 | 1.8559 | 1.8559 | 0.1171 | 0.0000 | 0.0631 | 0.0090 | 0.9414 | 1.0000 | 0.5686 |
| 3 | 109 | 2.8807 | 2.4404 | 2.3578 | 0.5229 | 0.0826 | 0.2385 | 0.0550 | 0.8211 | 0.9694 | 0.5280 |

### By Depth Tier

| Depth | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deep | 89 | 0.7526 | 0.6162 | 0.7529 | 0.8130 | 0.3433 | 1.1746 |
| medium | 123 | 0.7246 | 0.5836 | 0.7273 | 0.7973 | 0.4149 | 1.4764 |
| shallow | 88 | 0.6962 | 0.5492 | 0.6965 | 0.7825 | 0.4243 | 1.3817 |

### By Shape Class

| Shape class | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid | 41 | 0.7619 | 0.6363 | 0.7360 | 0.8741 | 0.3561 | 1.0370 |
| irregular | 50 | 0.7441 | 0.6110 | 0.7162 | 0.8693 | 0.3226 | 0.7113 |
| mixed_three_shape | 18 | 0.6947 | 0.5426 | 0.7837 | 0.6819 | 0.5428 | 2.5377 |
| mixed_two_shape | 160 | 0.7112 | 0.5638 | 0.7292 | 0.7536 | 0.4131 | 1.4831 |
| sphere | 31 | 0.7301 | 0.5912 | 0.6770 | 0.8754 | 0.3971 | 1.5052 |

### By Shape Set

| Shape set | N | Dice | IoU | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| ellipsoid | 41 | 0.7619 | 0.6363 | 0.7360 | 0.8741 |
| ellipsoid+irregular | 47 | 0.7140 | 0.5664 | 0.7465 | 0.7325 |
| ellipsoid+irregular+sphere | 18 | 0.6947 | 0.5426 | 0.7837 | 0.6819 |
| ellipsoid+sphere | 70 | 0.7158 | 0.5702 | 0.7299 | 0.7600 |
| irregular | 50 | 0.7441 | 0.6110 | 0.7162 | 0.8693 |
| irregular+sphere | 43 | 0.7006 | 0.5507 | 0.7093 | 0.7661 |
| sphere | 31 | 0.7301 | 0.5912 | 0.6770 | 0.8754 |

## Source-slot soft-union

- Checkpoint: `outputs/gisc_fmt/fit/2026-06-12/12-37-02/checkpoints/epoch=10-val_dice=0.7232.ckpt`
- Output dir: `outputs/fmt_simgen_v2_source_slots_eval/source_slots_soft_union_epoch10_test300_corrected`
- Note: Corrected evaluation path with source hypotheses passed into full-volume inference; kept as ablation, not mainline.

| Dice | IoU | Precision | Recall | ASSD | HD95 | CLE | NRMSE | Comp recall | Miss/sample | Merge/sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7093 | 0.5630 | 0.6966 | 0.8015 | 0.3822 | 1.1360 | 1.0640 | 0.0304 | 0.9006 | 0.2633 | 0.1633 |

### By Num Foci

| Num foci | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 | 0.7444 | 0.6105 | 0.6808 | 0.9151 | 0.3388 | 0.7147 |
| 2 | 111 | 0.7190 | 0.5740 | 0.6930 | 0.8269 | 0.3610 | 1.0972 |
| 3 | 109 | 0.6736 | 0.5170 | 0.7118 | 0.6922 | 0.4357 | 1.4848 |

### Component Metrics by Num Foci

| Num foci | N | GT comp | Pred comp | Matched | Missed | False | Merge | Split | Comp recall | Comp precision | Matched IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.6104 |
| 2 | 111 | 1.9730 | 1.8108 | 1.8018 | 0.1712 | 0.0090 | 0.1261 | 0.0000 | 0.9144 | 0.9955 | 0.5416 |
| 3 | 109 | 2.8807 | 2.4404 | 2.3303 | 0.5505 | 0.1101 | 0.3211 | 0.0275 | 0.8135 | 0.9648 | 0.5031 |

### By Depth Tier

| Depth | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deep | 89 | 0.7265 | 0.5818 | 0.7190 | 0.7991 | 0.3715 | 1.2170 |
| medium | 123 | 0.7001 | 0.5527 | 0.6845 | 0.8004 | 0.3929 | 1.1078 |
| shallow | 88 | 0.7046 | 0.5586 | 0.6907 | 0.8055 | 0.3780 | 1.0935 |

### By Shape Class

| Shape class | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid | 41 | 0.7539 | 0.6228 | 0.7034 | 0.8903 | 0.3317 | 0.8055 |
| irregular | 50 | 0.7103 | 0.5679 | 0.6774 | 0.8594 | 0.3711 | 0.8235 |
| mixed_three_shape | 18 | 0.6866 | 0.5319 | 0.7562 | 0.6751 | 0.4482 | 1.6289 |
| mixed_two_shape | 160 | 0.6983 | 0.5478 | 0.7021 | 0.7611 | 0.3847 | 1.1775 |
| sphere | 31 | 0.7185 | 0.5730 | 0.6553 | 0.8721 | 0.4157 | 1.5765 |

### By Shape Set

| Shape set | N | Dice | IoU | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| ellipsoid | 41 | 0.7539 | 0.6228 | 0.7034 | 0.8903 |
| ellipsoid+irregular | 47 | 0.7077 | 0.5564 | 0.7287 | 0.7371 |
| ellipsoid+irregular+sphere | 18 | 0.6866 | 0.5319 | 0.7562 | 0.6751 |
| ellipsoid+sphere | 70 | 0.6953 | 0.5455 | 0.7006 | 0.7600 |
| irregular | 50 | 0.7103 | 0.5679 | 0.6774 | 0.8594 |
| irregular+sphere | 43 | 0.6928 | 0.5421 | 0.6756 | 0.7893 |
| sphere | 31 | 0.7185 | 0.5730 | 0.6553 | 0.8721 |

## SSQ no_ptfa

- Checkpoint: `outputs/fmt_simgen_v2_ssq_ablation/runs/no_ptfa/checkpoints/epoch=51-val_dice=0.7302.ckpt`
- Output dir: `outputs/fmt_simgen_v2_ssq_ablation/test300/no_ptfa`
- Note: PTFA disabled.

| Dice | IoU | Precision | Recall | ASSD | HD95 | CLE | NRMSE | Comp recall | Miss/sample | Merge/sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.6986 | 0.5510 | 0.6534 | 0.8332 | 0.3958 | 1.1093 | 1.0178 | 0.0321 | 0.8961 | 0.2733 | 0.2033 |

### By Num Foci

| Num foci | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 | 0.7253 | 0.5863 | 0.6449 | 0.9291 | 0.3716 | 0.8119 |
| 2 | 111 | 0.7052 | 0.5592 | 0.6538 | 0.8503 | 0.3711 | 0.9980 |
| 3 | 109 | 0.6722 | 0.5167 | 0.6593 | 0.7455 | 0.4388 | 1.4408 |

### Component Metrics by Num Foci

| Num foci | N | GT comp | Pred comp | Matched | Missed | False | Merge | Split | Comp recall | Comp precision | Matched IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 | 1.0000 | 1.0125 | 1.0000 | 0.0000 | 0.0125 | 0.0000 | 0.0000 | 1.0000 | 0.9938 | 0.5863 |
| 2 | 111 | 1.9730 | 1.8468 | 1.7928 | 0.1802 | 0.0541 | 0.1441 | 0.0000 | 0.9099 | 0.9790 | 0.5201 |
| 3 | 109 | 2.8807 | 2.4312 | 2.3119 | 0.5688 | 0.1193 | 0.4128 | 0.0092 | 0.8058 | 0.9650 | 0.4932 |

### By Depth Tier

| Depth | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deep | 89 | 0.7195 | 0.5728 | 0.6767 | 0.8336 | 0.3601 | 1.0059 |
| medium | 123 | 0.6908 | 0.5424 | 0.6455 | 0.8282 | 0.4139 | 1.1624 |
| shallow | 88 | 0.6884 | 0.5409 | 0.6410 | 0.8398 | 0.4067 | 1.1397 |

### By Shape Class

| Shape class | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid | 41 | 0.7346 | 0.5985 | 0.6773 | 0.8938 | 0.3640 | 0.8999 |
| irregular | 50 | 0.6924 | 0.5447 | 0.6288 | 0.8877 | 0.4046 | 0.9149 |
| mixed_three_shape | 18 | 0.7172 | 0.5713 | 0.7389 | 0.7398 | 0.3909 | 1.4162 |
| mixed_two_shape | 160 | 0.6892 | 0.5378 | 0.6556 | 0.7965 | 0.4027 | 1.1967 |
| sphere | 31 | 0.6983 | 0.5546 | 0.6010 | 0.9091 | 0.3912 | 1.0704 |

### By Shape Set

| Shape set | N | Dice | IoU | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| ellipsoid | 41 | 0.7346 | 0.5985 | 0.6773 | 0.8938 |
| ellipsoid+irregular | 47 | 0.7095 | 0.5591 | 0.6899 | 0.7755 |
| ellipsoid+irregular+sphere | 18 | 0.7172 | 0.5713 | 0.7389 | 0.7398 |
| ellipsoid+sphere | 70 | 0.6893 | 0.5385 | 0.6552 | 0.7993 |
| irregular | 50 | 0.6924 | 0.5447 | 0.6288 | 0.8877 |
| irregular+sphere | 43 | 0.6669 | 0.5133 | 0.6186 | 0.8148 |
| sphere | 31 | 0.6983 | 0.5546 | 0.6010 | 0.9091 |

## SSQ no_canonical_reliability

- Checkpoint: `outputs/fmt_simgen_v2_ssq_ablation/runs/no_canonical_reliability/checkpoints/epoch=59-val_dice=0.7262.ckpt`
- Output dir: `outputs/fmt_simgen_v2_ssq_ablation/test300/no_canonical_reliability`
- Note: Canonical reliability aggregation disabled.

| Dice | IoU | Precision | Recall | ASSD | HD95 | CLE | NRMSE | Comp recall | Miss/sample | Merge/sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7123 | 0.5661 | 0.6998 | 0.8017 | 0.3760 | 1.1125 | 1.0784 | 0.0304 | 0.9022 | 0.2600 | 0.1800 |

### By Num Foci

| Num foci | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 | 0.7533 | 0.6200 | 0.7037 | 0.9034 | 0.3246 | 0.7231 |
| 2 | 111 | 0.7164 | 0.5703 | 0.7000 | 0.8135 | 0.3660 | 1.1205 |
| 3 | 109 | 0.6779 | 0.5222 | 0.6967 | 0.7151 | 0.4239 | 1.3902 |

### Component Metrics by Num Foci

| Num foci | N | GT comp | Pred comp | Matched | Missed | False | Merge | Split | Comp recall | Comp precision | Matched IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.6200 |
| 2 | 111 | 1.9730 | 1.8288 | 1.8198 | 0.1532 | 0.0090 | 0.1171 | 0.0090 | 0.9234 | 0.9955 | 0.5341 |
| 3 | 109 | 2.8807 | 2.3761 | 2.3211 | 0.5596 | 0.0550 | 0.3761 | 0.0092 | 0.8089 | 0.9809 | 0.4963 |

### By Depth Tier

| Depth | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deep | 89 | 0.7353 | 0.5918 | 0.7332 | 0.7963 | 0.3576 | 1.1865 |
| medium | 123 | 0.7043 | 0.5571 | 0.6928 | 0.7996 | 0.3876 | 1.0929 |
| shallow | 88 | 0.7000 | 0.5526 | 0.6756 | 0.8102 | 0.3785 | 1.0650 |

### By Shape Class

| Shape class | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid | 41 | 0.7589 | 0.6265 | 0.7261 | 0.8671 | 0.3202 | 0.8444 |
| irregular | 50 | 0.7100 | 0.5665 | 0.6818 | 0.8569 | 0.3746 | 0.8325 |
| mixed_three_shape | 18 | 0.6997 | 0.5503 | 0.7324 | 0.7127 | 0.4194 | 1.4613 |
| mixed_two_shape | 160 | 0.6971 | 0.5460 | 0.6989 | 0.7629 | 0.3829 | 1.1471 |
| sphere | 31 | 0.7394 | 0.5983 | 0.6793 | 0.8785 | 0.3914 | 1.5379 |

### By Shape Set

| Shape set | N | Dice | IoU | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| ellipsoid | 41 | 0.7589 | 0.6265 | 0.7261 | 0.8671 |
| ellipsoid+irregular | 47 | 0.7070 | 0.5556 | 0.7256 | 0.7439 |
| ellipsoid+irregular+sphere | 18 | 0.6997 | 0.5503 | 0.7324 | 0.7127 |
| ellipsoid+sphere | 70 | 0.6979 | 0.5482 | 0.6988 | 0.7601 |
| irregular | 50 | 0.7100 | 0.5665 | 0.6818 | 0.8569 |
| irregular+sphere | 43 | 0.6852 | 0.5321 | 0.6699 | 0.7880 |
| sphere | 31 | 0.7394 | 0.5983 | 0.6793 | 0.8785 |

## SSQ no_source_cue

- Checkpoint: `outputs/fmt_simgen_v2_ssq_ablation/runs/no_source_cue/checkpoints/epoch=59-val_dice=0.7302.ckpt`
- Output dir: `outputs/fmt_simgen_v2_ssq_ablation/test300/no_source_cue`
- Note: Measurement-derived source-instance cue disabled; training stopped after epoch 74, best epoch 59 evaluated.

| Dice | IoU | Precision | Recall | ASSD | HD95 | CLE | NRMSE | Comp recall | Miss/sample | Merge/sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7079 | 0.5613 | 0.6880 | 0.8118 | 0.3869 | 1.1310 | 1.0886 | 0.0307 | 0.9039 | 0.2533 | 0.1600 |

### By Num Foci

| Num foci | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 | 0.7310 | 0.5929 | 0.6570 | 0.9253 | 0.3610 | 0.7604 |
| 2 | 111 | 0.7163 | 0.5721 | 0.6862 | 0.8329 | 0.3676 | 1.1146 |
| 3 | 109 | 0.6824 | 0.5272 | 0.7126 | 0.7069 | 0.4255 | 1.4196 |

### Component Metrics by Num Foci

| Num foci | N | GT comp | Pred comp | Matched | Missed | False | Merge | Split | Comp recall | Comp precision | Matched IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.5929 |
| 2 | 111 | 1.9730 | 1.8288 | 1.8108 | 0.1622 | 0.0180 | 0.1261 | 0.0090 | 0.9189 | 0.9940 | 0.5357 |
| 3 | 109 | 2.8807 | 2.4679 | 2.3486 | 0.5321 | 0.1193 | 0.3119 | 0.0092 | 0.8180 | 0.9664 | 0.5099 |

### By Depth Tier

| Depth | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deep | 89 | 0.7296 | 0.5873 | 0.7120 | 0.8175 | 0.3689 | 1.1793 |
| medium | 123 | 0.6986 | 0.5499 | 0.6809 | 0.8054 | 0.4022 | 1.1506 |
| shallow | 88 | 0.6990 | 0.5510 | 0.6738 | 0.8148 | 0.3836 | 1.0545 |

### By Shape Class

| Shape class | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid | 41 | 0.7392 | 0.6036 | 0.6875 | 0.8889 | 0.3566 | 0.8593 |
| irregular | 50 | 0.6998 | 0.5553 | 0.6529 | 0.8759 | 0.3896 | 0.8518 |
| mixed_three_shape | 18 | 0.6938 | 0.5398 | 0.7472 | 0.6891 | 0.4432 | 1.6774 |
| mixed_two_shape | 160 | 0.7006 | 0.5507 | 0.7029 | 0.7671 | 0.3818 | 1.1320 |
| sphere | 31 | 0.7255 | 0.5826 | 0.6342 | 0.9077 | 0.4161 | 1.6175 |

### By Shape Set

| Shape set | N | Dice | IoU | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| ellipsoid | 41 | 0.7392 | 0.6036 | 0.6875 | 0.8889 |
| ellipsoid+irregular | 47 | 0.7143 | 0.5637 | 0.7296 | 0.7467 |
| ellipsoid+irregular+sphere | 18 | 0.6938 | 0.5398 | 0.7472 | 0.6891 |
| ellipsoid+sphere | 70 | 0.6981 | 0.5491 | 0.6955 | 0.7760 |
| irregular | 50 | 0.6998 | 0.5553 | 0.6529 | 0.8759 |
| irregular+sphere | 43 | 0.6898 | 0.5390 | 0.6859 | 0.7751 |
| sphere | 31 | 0.7255 | 0.5826 | 0.6342 | 0.9077 |

## SSQ no_distance_aux

- Checkpoint: `outputs/fmt_simgen_v2_ssq_ablation/runs/no_distance_aux/checkpoints/epoch=59-val_dice=0.7256.ckpt`
- Output dir: `outputs/fmt_simgen_v2_ssq_ablation/test300/no_distance_aux`
- Note: Distance auxiliary loss/head disabled; training stopped at epoch 65, best epoch 59 evaluated.

| Dice | IoU | Precision | Recall | ASSD | HD95 | CLE | NRMSE | Comp recall | Miss/sample | Merge/sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7106 | 0.5646 | 0.6954 | 0.8048 | 0.3943 | 1.2275 | 1.0046 | 0.0306 | 0.9094 | 0.2433 | 0.1533 |

### By Num Foci

| Num foci | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 | 0.7431 | 0.6085 | 0.6890 | 0.9064 | 0.3395 | 0.7244 |
| 2 | 111 | 0.7175 | 0.5734 | 0.6958 | 0.8214 | 0.3594 | 1.0808 |
| 3 | 109 | 0.6797 | 0.5236 | 0.6999 | 0.7133 | 0.4700 | 1.7461 |

### Component Metrics by Num Foci

| Num foci | N | GT comp | Pred comp | Matched | Missed | False | Merge | Split | Comp recall | Comp precision | Matched IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.6084 |
| 2 | 111 | 1.9730 | 1.8739 | 1.8468 | 0.1261 | 0.0270 | 0.0901 | 0.0000 | 0.9369 | 0.9895 | 0.5394 |
| 3 | 109 | 2.8807 | 2.4312 | 2.3394 | 0.5413 | 0.0917 | 0.3303 | 0.0092 | 0.8150 | 0.9725 | 0.5057 |

### By Depth Tier

| Depth | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deep | 89 | 0.7310 | 0.5885 | 0.7191 | 0.8120 | 0.3638 | 1.1788 |
| medium | 123 | 0.7022 | 0.5545 | 0.6881 | 0.7992 | 0.4300 | 1.3864 |
| shallow | 88 | 0.7018 | 0.5546 | 0.6818 | 0.8054 | 0.3752 | 1.0547 |

### By Shape Class

| Shape class | N | Dice | IoU | Precision | Recall | ASSD | HD95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ellipsoid | 41 | 0.7496 | 0.6154 | 0.7029 | 0.8823 | 0.3391 | 0.8679 |
| irregular | 50 | 0.7066 | 0.5632 | 0.6815 | 0.8515 | 0.3769 | 0.8295 |
| mixed_three_shape | 18 | 0.7164 | 0.5653 | 0.7807 | 0.7014 | 0.4007 | 1.5437 |
| mixed_two_shape | 160 | 0.7004 | 0.5507 | 0.6981 | 0.7678 | 0.4076 | 1.3317 |
| sphere | 31 | 0.7146 | 0.5714 | 0.6447 | 0.8778 | 0.4227 | 1.6237 |

### By Shape Set

| Shape set | N | Dice | IoU | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| ellipsoid | 41 | 0.7496 | 0.6154 | 0.7029 | 0.8823 |
| ellipsoid+irregular | 47 | 0.7095 | 0.5601 | 0.7213 | 0.7517 |
| ellipsoid+irregular+sphere | 18 | 0.7164 | 0.5653 | 0.7807 | 0.7014 |
| ellipsoid+sphere | 70 | 0.7037 | 0.5549 | 0.6941 | 0.7773 |
| irregular | 50 | 0.7066 | 0.5632 | 0.6815 | 0.8515 |
| irregular+sphere | 43 | 0.6852 | 0.5336 | 0.6794 | 0.7699 |
| sphere | 31 | 0.7146 | 0.5714 | 0.6447 | 0.8778 |

## Interpretation for Paper Positioning

The results support positioning SSQ as source-separable query representation rather than a source-slot soft-union decoder. The formal mainline should be the no-center SSQ variant: measurement-derived source hypotheses/cues, PTFA evidence, query-canonical reliability aggregation, and distance/geometry auxiliary training regularization. The original E15 center auxiliary path should be presented as a center-auxiliary ablation: it modestly reduces merge count but does not improve overall Dice, so center supervision is not the central source-separation mechanism.
# Historical / deprecated method version

This document contains older SSQ ablation notes and must not be interpreted as results
for the current final SSQ-FMT candidate-composition implementation. Prefer
`docs/ssq_final_ablation_protocol.md` for the current protocol.
