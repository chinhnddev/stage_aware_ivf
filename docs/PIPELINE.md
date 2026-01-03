# IVF Stage-Aware Morphology Pipeline

Goal: infer embryo quality (good/poor) from morphology with stage-aware conditioning.

Overview (10 lines):
1) EXP-1 learns morphology (EXP/ICM/TE) from images.
2) EXP-2 learns stage-aware representation (cleavage/morula/blastocyst).
3) EXP-3 stabilizes by joint fine-tuning (morph + stage) with a freeze schedule.
4) EXP-4 trains only the quality head (encoder+morph+stage frozen).
5) EXP-4 uses Gardner-derived proxy labels (exp/icm/te) when true quality is unavailable.
6) EXP-5 evaluates cross-domain on Hung Vuong with zero-shot and calibrated modes.
7) Zero-shot uses the source (EXP-4 val) threshold only.
8) Calibrated mode tunes threshold/temperature on HV-val only, reports on HV-test.
9) External eval never derives labels from morphology at inference time.
10) Optional EXP-M: rule-based quality from predicted morphology (analysis only).

ASCII diagram:
```
Image --> Encoder --> Morph Heads (EXP/ICM/TE)
                 \-> Stage Head (cleavage/morula/blastocyst)
                      \-> Quality Head (conditioned on morph + stage)
```

Notes:
- EXP-3 is a training stage only (not a standalone results row).
- External labels are taken from external CSV quality labels; Gardner rules are not used at inference.
- EXP-4 ablations: use `configs/experiment/exp4_morph_stage.yaml`, `exp4_stage_only.yaml`, or `exp4_none.yaml`.
