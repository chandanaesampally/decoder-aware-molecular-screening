# Decoder-Aware Molecular Screening

A dashboard that connects learned decoder-triage policies for surface-code error correction to VQE ground-state energy estimation on small molecules. The application shows how decoder quality propagates through to chemical accuracy on final energy estimates.

## Overview

**The problem:** In surface-code memory experiments, Minimum-Weight Perfect Matching (MWPM) is fast but sometimes inaccurate. Belief Propagation with Ordered Statistics Decoding (BP+OSD) is accurate but slow.

**Our solution:** A triage policy identifies shots likely mis-decoded by MWPM and escalates only those to BP+OSD — trading latency for accuracy. We train a Graph Attention Network to perform this triage and compare it against two baselines.

**The pipeline:**

1. **Stage 1** (`generate_data.py`): Simulates a distance-5 surface-code memory experiment
2. **Stage 2** (`decoder_triage_pipeline.py`): Trains the triage GNN and evaluates all strategies
3. **Stage 3** (`app.py`): Maps decoder performance to VQE accuracy on small molecules

## Quick Start

```bash
# Clone and enter project
git clone <repository-url>
cd decoder_triage_project

# Set up environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python generate_data.py
python decoder_triage_pipeline.py
python app.py

# Open http://127.0.0.1:8060/