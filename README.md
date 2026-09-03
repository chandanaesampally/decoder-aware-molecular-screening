# Decoder-Aware Molecular Screening

A dashboard that connects learned decoder-triage policies for surface-code error correction to VQE ground-state energy estimation on small molecules. The application shows how decoder quality propagates through to chemical accuracy on final energy estimates.

## Overview

**The problem:** In surface-code memory experiments, Minimum-Weight Perfect Matching (MWPM) is fast but sometimes inaccurate. Belief Propagation with Ordered Statistics Decoding (BP+OSD) is accurate but slow.

**Our solution:** A triage policy identifies shots likely mis-decoded by MWPM and escalates only those to BP+OSD — trading latency for accuracy. We train a Graph Attention Network to perform this triage and compare it against two baselines.

**The pipeline:**

1. **Stage 1** (`generate_data.py`): Simulates a distance-5 surface-code memory experiment
2. **Stage 2** (`decoder_triage_pipeline.py`): Trains the triage GNN and evaluates all strategies
3. **Stage 3** (`app.py`): Maps decoder performance to VQE accuracy on small molecules

## Repository

```
https://github.com/chandanaesampally/decoder-aware-molecular-screening
```

## Quick Start

```bash
## Requirements
- **Python:** 3.10 or 3.11 (3.12+ may have compatibility issues with `torch-geometric`)
- **OS:** Linux, macOS, or Windows
- **GPU:** Optional (automatically falls back to CPU)
- **RAM:** 8GB+ recommended for Stage 2
- **Storage:** ~500MB for generated data and models

## Detailed Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/chandanaesampally/decoder-aware-molecular-screening.git
cd decoder-aware-molecular-screening
```

### 2. Set Up Virtual Environment

**Using VSCode (Recommended):**

1. Open the project folder in VSCode
2. Press `Ctrl+Shift+P` (Cmd+Shift+P on Mac) to open command palette
3. Type "Python: Create Environment" and select it
4. Choose "Venv" as the environment type
5. Select Python 3.10 or 3.11 as the interpreter
6. VSCode will automatically create `venv/` and activate it
7. Press `Ctrl+Shift+P` again and select "Python: Select Interpreter"
8. Choose the one that says `./venv/bin/python` or `./venv/Scripts/python.exe`

**Using Terminal:**

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**Verify Environment:**
```bash
# Should show the venv path
python -c "import sys; print(sys.executable)"

# Should show (venv) in your terminal prompt
```

### 3. Create Project Folder Structure

The scripts automatically create necessary folders, but you can pre-create them:

```bash
# Linux/Mac
mkdir -p outputs/edge_reweight_data
mkdir -p outputs/triage_models
mkdir -p outputs/triage_results
mkdir -p outputs/screening_cache

# Windows (PowerShell)
New-Item -ItemType Directory -Force -Path outputs/edge_reweight_data
New-Item -ItemType Directory -Force -Path outputs/triage_models
New-Item -ItemType Directory -Force -Path outputs/triage_results
New-Item -ItemType Directory -Force -Path outputs/screening_cache
```

**Final folder structure:**
```
decoder-aware-molecular-screening/
├── generate_data.py              # Stage 1
├── decoder_triage_pipeline.py    # Stage 2
├── app.py                        # Stage 3
├── requirements.txt
├── outputs/
│   ├── edge_reweight_data/       # Stage 1 output
│   ├── triage_models/            # Stage 2: models & scores
│   ├── triage_results/           # Stage 2: plots
│   └── screening_cache/          # Stage 3: cached VQE results
└── venv/                         # Virtual environment
```

### 4. Install Dependencies

**All dependencies in one command:**
```bash
pip install -r requirements.txt
```

**Alternative: Install grouped by stage:**

| Stage | Packages |
|-------|----------|
| **Core (Stage 1-2)** | `numpy`, `stim`, `pymatching` |
| **Stage 2** | `torch`, `torch-geometric`, `scikit-learn`, `matplotlib`, `stimbposd` |
| **Stage 3** | `dash`, `plotly`, `pennylane`, `rdkit` |

```bash
# Core
pip install numpy stim pymatching

# Stage 2 (CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric scikit-learn matplotlib stimbposd

# Stage 3
pip install dash plotly pennylane rdkit
```

**GPU Support (Optional):**
```bash
# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 5. Verify Installation

```bash
python -c "import numpy, stim, pymatching, torch, torch_geometric, dash, pennylane; print('All packages imported successfully')"
```

## Running the Pipeline

### Stage 1: Data Generation
```bash
python generate_data.py
```

**What it does:**
- Simulates distance-5 surface-code memory experiment
- Generates 100,000 training shots and 50,000 test shots at `p=0.005`
- Saves detector coordinates and syndrome data

**Output:**
- `outputs/edge_reweight_data/dem_structure_Z_p0.0050.json`
- `outputs/edge_reweight_data/train_Z_p0.0050.npz`
- `outputs/edge_reweight_data/test_Z_p0.0050.npz`

**Expected output:**
```
Building circuit: basis=Z, p=0.005, distance=5, rounds=5
Saved structure -> outputs/edge_reweight_data/dem_structure_Z_p0.0050.json (detectors)
Sampling 100,000 training shots...
Sampling 50,000 test shots...
[sanity check] plain MWPM on this test set: LER ≈ 0.01-0.02
```

### Stage 2: Triage Pipeline
```bash
python decoder_triage_pipeline.py
```

**What it does:**
- Trains a 3-layer Graph Attention Network (15 epochs)
- Computes baseline scores (syndrome weight, classical ensemble)
- Evaluates all strategies at escalation rates 0-30%
- Measures real per-shot latencies
- Saves model and scores for Stage 3

**⚠️ This is the heavy stage:** Expect longer run time, especially on CPU.

**Output:**
- `outputs/triage_models/triage_gnn_state.pt` (trained model)
- `outputs/triage_models/scores_test.npz` (input to Stage 3)
- `outputs/triage_results/roc_curve.png`
- `outputs/triage_results/ler_tradeoff.png`

**Expected output:**
```
=== QUANTUM OS ROUTING ENGINE: End-to-End Evaluation ===
1. Loading Simulated Dataset...
2. Generating Baseline Decodes (MWPM & BP+OSD)...
  MWPM LER -> Train: ~0.01300 | Test: ~0.01250
  BP+OSD LER -> Test: ~0.00150
3. Computing Classical Baselines...
4. Training Proposed Model: Spatiotemporal GNN...
  Epoch 01 | BCE Loss: 0.3245 | Time: 45s
  ...
5. Extracting GNN Predictions...
6. Measuring Real Per-Shot Decoder Latency...
  MWPM:   10.23 us/shot
  GNN:    12.45 us/shot
  BP+OSD: 245.67 us/shot
...
```

### Stage 3: Dashboard
```bash
python app.py
```

**What it does:**
- Launches interactive Dash web application
- Reads `scores_test.npz` from Stage 2
- Maps each strategy's LER to VQE noise level
- Runs VQE on 14 small molecules with zero-noise extrapolation
- Reports chemical accuracy tiers

**Open in browser:** http://127.0.0.1:8060/

**Expected output:**
```
========================================================================
 Decoder-aware molecular screening
   scores: outputs/triage_models/scores_test.npz (found)
   pennylane: available
========================================================================
Dash is running on http://127.0.0.1:8060/
```

## Project Structure Details

### File Dependencies
```
generate_data.py
    ↓ (writes)
outputs/edge_reweight_data/
    ├── dem_structure_Z_p0.0050.json
    ├── train_Z_p0.0050.npz
    └── test_Z_p0.0050.npz
    ↓ (read by)
decoder_triage_pipeline.py
    ↓ (writes)
outputs/triage_models/
    ├── triage_gnn_state.pt
    └── scores_test.npz
    ↓ (read by)
app.py
    ↓ (writes)
outputs/screening_cache/
    └── *.json (cached VQE results)
```

### Configuration Constants

**Important:** `BASIS`, `P`, `DISTANCE`, `ROUNDS` must match between `generate_data.py` and `decoder_triage_pipeline.py`:

| File | BASIS | P | DISTANCE | ROUNDS |
|------|-------|---|----------|--------|
| `generate_data.py` | `"Z"` | `0.005` | `5` | `5` |
| `decoder_triage_pipeline.py` | `"Z"` | `0.005` | `5` | `5` |

If you change these, update both files.

## Molecule Library

**14 VQE-simulated molecules (4-8 qubits):**
H₂, HeH⁺, LiH, HF, BeH₂, N₂, CO, HCN, CH₂O, C₂H₂, CO₂, H₂O, NH₃, CH₄

**4 Drug gallery molecules (structure only):**
Aspirin, Ibuprofen, Caffeine, Paracetamol
- Shown for structure/qubit reasoning
- NOT run through VQE (active spaces too large)
- Labeled "(preview only)" in dropdown

## requirements.txt

Create a `requirements.txt` file with the following content:

```
# Core dependencies (Stage 1 & 2)
numpy==1.26.4
stim==1.14.0
pymatching==2.1.0

# Stage 2: Triage Pipeline
torch==2.3.1
torch-geometric==2.5.3
scikit-learn==1.5.0
matplotlib==3.9.0
stimbposd==1.1.0

# Stage 3: Dashboard
dash==2.17.1
plotly==5.24.1
pennylane==0.38.0
rdkit==2024.3.5
```

**Note:** For PyTorch, you may need to use the appropriate index URL for your system:
- CPU: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- CUDA 12.1: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- CUDA 11.8: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `torch-geometric` fails to install | Install `torch` first, then `torch-geometric` |
| Port 8060 already in use | Set `PORT` environment variable: `PORT=8061 python app.py` |
| Missing `scores_test.npz` | Run Stage 2 successfully first |
| Molecule diagrams not rendering | Install `rdkit`: `pip install rdkit` |
| VQE not running | Install `pennylane`: `pip install pennylane` |
| Dash dropdown unreadable | CSS fixes included in `app.py`'s index string |
| HeH+ diagram shows boxes | Expected - RDKit cannot represent bare noble-gas cation |
| `stimbposd` import error | Install: `pip install stimbposd` |
| Python 3.12 compatibility | Use Python 3.10 or 3.11 instead |

### Performance Notes

- **Stage 2 on CPU:** Can take 30-60 minutes (depends on CPU speed)
- **Stage 2 on GPU:** Can take 5-15 minutes
- **H₂O VQE:** ~26 ansatz parameters → 1-3 minutes first run (cached after)
- **Most molecules:** ~3-6 parameters → seconds to run

### VSCode Tips

1. **Select Interpreter:** `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose `venv`
2. **Run Script:** Right-click file → "Run Python File in Terminal"
3. **Debug:** Set breakpoints → F5 → Select "Python File"
4. **Terminal:** Ctrl+` (backtick) to open integrated terminal
5. **Git Integration:** Built-in source control panel

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
