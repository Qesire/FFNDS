# Anchor5 单DEV配对分数

仅为本轮同配置三臂的原值与差值，不是等质/泛化/显著性结论。

| 指标 | Dense | GPU FP32 | GPU BF16 | FP32−Dense | BF16−Dense | BF16−FP32 | 单位 |
|---|---:|---:|---:|---:|---:|---:|---|
| subject_consistency | 0.95169368 | 0.891347652 | 0.901639961 | -0.0603460275 | -0.0500537183 | 0.0102923092 | normalized proxy score (higher proxy value) |
| background_consistency | 0.959179688 | 0.951882935 | 0.948550415 | -0.00729675293 | -0.0106292725 | -0.00333251953 | normalized proxy score (higher proxy value) |
| motion_smoothness | 0.980620946 | 0.962235824 | 0.960514198 | -0.0183851218 | -0.0201067478 | -0.00172162597 | normalized proxy score (higher proxy value) |
| dynamic_degree | 0 | 0 | 0 | 0 | 0 | 0 | official motion threshold {0,1}; not quality direction |
| aesthetic_quality | 0.464501172 | 0.410617173 | 0.396446526 | -0.0538839996 | -0.0680546463 | -0.0141706467 | aesthetic regressor /10 (not explicitly clamped) |
| imaging_quality | 74.654646 | 77.0526927 | 77.5536084 | 2.39804671 | 2.89896233 | 0.500915622 | MUSIQ raw score (nominal 0–100) |
| i2v_subject | 0.983809507 | 0.880145446 | 0.88506239 | -0.103664061 | -0.098747117 | 0.00491694443 | normalized proxy score (higher proxy value) |
| i2v_background | 0.989141508 | 0.945768535 | 0.950012558 | -0.0433729733 | -0.0391289502 | 0.00424402304 | normalized proxy score (higher proxy value) |
| camera_motion | 0 | 0 | 0 | 0 | 0 | 0 | target-label membership {0,1} |
| temporal_flickering | 0.957255004 | 0.939532089 | 0.93670676 | -0.0177229152 | -0.0205482446 | -0.00282532935 | normalized proxy score (higher proxy value) |
