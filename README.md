# Decoder-Aware Molecular Screening

A dashboard that connects two independent stages of a NISQ chemistry
pipeline: a learned decoder-triage policy for surface-code error
correction, and a VQE ground-state energy estimator for a small
molecule library. The application shows how the choice of decoding
strategy propagates through to chemical accuracy on the final energy
estimate.

## Background

Surface-code memory experiments are decoded per shot. Minimum-weight
perfect matching (MWPM) is fast but not always accurate; belief
propagation with ordered statistics decoding (BP+OSD) is more accurate
but slower. A triage policy identifies which shots are likely to be
mis-decoded by MWPM and escalates only those to BP+OSD, trading latency
against logical error rate.

`decoder_triage_pipeline.py` trains a graph attention network on
simulated distance-5 rotated surface code data to perform this triage,
and compares it against two baselines: raw syndrome weight and a
matching-ensemble disagreement score. Its output — measured logical
error rates and latencies for each strategy, at a sweep of escalation
percentages — is saved to `outputs/triage_models/scores_test.npz`.

`app.py` reads that file and uses the resulting per-strategy
(LER, latency) pairs as the noise level for a VQE simulation on a
library of small molecules (H2, LiH, H2O, HeH+, HF, N2, HCN, CH2O,
C2H2, CO2). Each strategy's decoder quality determines the simulated
circuit noise; the resulting VQE energy error, before and after zero-noise
extrapolation, is reported against the exact (correct-sector) ground
state energy for that molecule.

## Repository contents

```
app.py                              Dash application (this is what runs)
requirements.txt                    Python dependencies for app.py
Dockerfile                          Container build for deployment
.dockerignore
.gitignore
outputs/triage_models/scores_test.npz   Output of decoder_triage_pipeline.py
```

`decoder_triage_pipeline.py` is not part of this repository. It is a
separate, GPU-preferred training script with its own dependencies
(PyTorch, PyTorch Geometric, PyMatching, Stim, stimbposd) and is run
once, offline, to produce `scores_test.npz`. Only that output file is
required for the dashboard to run.

## Requirements

- Python 3.11
- dash, plotly, numpy
- pennylane (VQE simulation)
- rdkit (2D molecule structure rendering)

See `requirements.txt` for pinned versions.

## Running locally

```
pip install -r requirements.txt
python app.py
```

The app reads `outputs/triage_models/scores_test.npz` relative to the
working directory it is started from. If that file is not present, the
dashboard still loads but reports it as missing and cannot run a
screening.

The app listens on `$PORT` if set, otherwise `8060`.

## Deployment

The included `Dockerfile` builds a container that installs
`requirements.txt` and copies `app.py` and `outputs/` into the image.
On a platform that provisions `$PORT` at runtime (Render, for example),
no additional configuration is needed: connect the repository, select
Docker as the environment, and deploy.

## Data dependency

`scores_test.npz` contains measured (not fabricated) test-set decoder
scores, labels, and latencies, plus a full escalation-rate sweep, for
five strategies: MWPM only, syndrome-weight triage, matching-ensemble
triage, GNN triage, and BP+OSD only. Regenerating it requires running
`decoder_triage_pipeline.py` against simulated surface-code data.

## Known limitations

The molecule library is restricted to 4-8 qubit active spaces so that
VQE remains classically simulable; these are standard benchmark
molecules, not drug-scale candidates. Circuit noise is injected as a
flat depolarizing channel sized to each strategy's measured logical
error rate — a proxy for how decoder quality affects a downstream
computation, not a full simulation of a fault-tolerant chemistry
circuit under an actual surface-code and decoder stack. Zero-noise
extrapolation is applied in simulation, where noise scaling is exact;
on real hardware, noise scaling is done by gate folding and typically
recovers less error than the idealized result shown here.
