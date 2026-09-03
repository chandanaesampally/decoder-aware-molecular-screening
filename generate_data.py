"""
generate_data.py

STEP 1 of 3 in the pipeline:

    python generate_data.py              (this file -- run first)
    python decoder_triage_pipeline.py    (trains the triage GNN)
    python app1.py                       (dashboard)

WHAT THIS PRODUCES, AND WHY THESE EXACT FILES:
decoder_triage_pipeline.py loads three things before it does anything else:

    outputs/edge_reweight_data/dem_structure_Z_p0.0050.json
        -> needs only "coords" and "basis_flag" (verified against
           decoder_triage_pipeline.py's actual read: 
           `coords, basis_flag = np.array(s["coords"]), s["basis_flag"]`)
    outputs/edge_reweight_data/train_Z_p0.0050.npz
        -> needs "detection_events" and "observable_flips", >= 100,000 shots
    outputs/edge_reweight_data/test_Z_p0.0050.npz
        -> needs "detection_events" and "observable_flips", >= 50,000 shots

Nothing else is read from these files downstream -- no edge_fired array,
no DEM edge list -- so this script only generates what's actually used.
decoder_triage_pipeline.py builds its own DEM/MWPM/BP+OSD internally from
the circuit parameters (BASIS, P, DISTANCE, ROUNDS) directly, so as long
as those four match here (they do: Z, 0.005, 5, 5), everything downstream
is fully compatible. This script does not read or depend on any of the
old iteration files (01_baseline.py, 02_generate_training_data.py, or the
margin-correction test5_*.py files) -- it's a clean, minimal extraction of
just the data-generation logic decoder_triage_pipeline.py actually needs.

Safe to re-run: if sufficient data already exists on disk, this exits
immediately without resampling.
"""

import os
import json
import numpy as np
import stim
import pymatching

# ----------------------------------------------------------------------
# Config -- MUST match decoder_triage_pipeline.py's BASIS/P/DISTANCE/ROUNDS
# ----------------------------------------------------------------------

BASIS    = "Z"
P        = 0.005
DISTANCE = 5
ROUNDS   = DISTANCE

DATA_ROOT      = os.path.join("outputs", "edge_reweight_data")
STRUCTURE_PATH = os.path.join(DATA_ROOT, f"dem_structure_{BASIS}_p{P:.4f}.json")
TRAIN_PATH     = os.path.join(DATA_ROOT, f"train_{BASIS}_p{P:.4f}.npz")
TEST_PATH      = os.path.join(DATA_ROOT, f"test_{BASIS}_p{P:.4f}.npz")

# Must be >= decoder_triage_pipeline.py's TRAIN_SUBSET_SHOTS (100_000) and
# TEST_SUBSET_SHOTS (50_000) -- that script does
# `n_train = min(TRAIN_SUBSET_SHOTS, actual_shots_in_file)`, so generating
# extra here is harmless, generating fewer would silently under-supply it.
TRAIN_SHOTS = 100_000
TEST_SHOTS  = 50_000

TRAIN_SEED = 170_005
TEST_SEED  = 190_005


# ----------------------------------------------------------------------
# Circuit
# ----------------------------------------------------------------------

def build_circuit(basis: str, p: float) -> stim.Circuit:
    task = f"surface_code:rotated_memory_{basis.lower()}"
    return stim.Circuit.generated(
        task,
        distance=DISTANCE,
        rounds=ROUNDS,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


def _shots_in(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    try:
        with np.load(path) as d:
            return int(d["detection_events"].shape[0])
    except Exception:
        return 0


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    os.makedirs(DATA_ROOT, exist_ok=True)

    have_train     = _shots_in(TRAIN_PATH) >= TRAIN_SHOTS
    have_test      = _shots_in(TEST_PATH)  >= TEST_SHOTS
    have_structure = os.path.isfile(STRUCTURE_PATH)

    if have_train and have_test and have_structure:
        print("Sufficient data already present -- nothing to regenerate.")
        print(f"  {STRUCTURE_PATH}")
        print(f"  {TRAIN_PATH}  ({_shots_in(TRAIN_PATH):,} shots)")
        print(f"  {TEST_PATH}  ({_shots_in(TEST_PATH):,} shots)")
        print("\nNext: python decoder_triage_pipeline.py")
        return

    print(f"Building circuit: basis={BASIS}, p={P}, distance={DISTANCE}, rounds={ROUNDS}")
    circuit = build_circuit(BASIS, P)
    dem     = circuit.detector_error_model(decompose_errors=True)

    # ---- Structure file ----
    coord_dict = circuit.get_detector_coordinates()
    coords     = np.zeros((circuit.num_detectors, 3), dtype=np.float32)
    for idx, xyz in coord_dict.items():
        padded      = list(xyz) + [0.0] * (3 - len(xyz))
        coords[idx] = padded[:3]
    basis_flag = 1.0 if BASIS == "X" else 0.0

    with open(STRUCTURE_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "basis": BASIS, "p": P, "distance": DISTANCE, "rounds": ROUNDS,
            "num_detectors": circuit.num_detectors,
            "coords": coords.tolist(),
            "basis_flag": basis_flag,
        }, f)
    print(f"Saved structure -> {STRUCTURE_PATH}  ({circuit.num_detectors} detectors)")

    # ---- Train / test shots ----
    def sample(shots: int, seed: int):
        sampler   = circuit.compile_detector_sampler(seed=seed)
        det, obs  = sampler.sample(shots=shots, separate_observables=True)
        return det, obs

    print(f"Sampling {TRAIN_SHOTS:,} training shots (seed={TRAIN_SEED})...")
    train_det, train_obs = sample(TRAIN_SHOTS, TRAIN_SEED)
    np.savez_compressed(
        TRAIN_PATH,
        detection_events=train_det.astype(np.bool_),
        observable_flips=train_obs.astype(np.bool_),
    )
    print(f"Saved -> {TRAIN_PATH}")

    print(f"Sampling {TEST_SHOTS:,} test shots (seed={TEST_SEED})...")
    test_det, test_obs = sample(TEST_SHOTS, TEST_SEED)
    np.savez_compressed(
        TEST_PATH,
        detection_events=test_det.astype(np.bool_),
        observable_flips=test_obs.astype(np.bool_),
    )
    print(f"Saved -> {TEST_PATH}")

    # ---- Sanity check: a real MWPM number before you invest time training ----
    matcher = pymatching.Matching.from_detector_error_model(dem)
    pred    = matcher.decode_batch(test_det)
    fails   = int(np.any(pred != test_obs, axis=1).sum())
    ler     = fails / TEST_SHOTS
    print(f"\n[sanity check] plain MWPM on this test set: {fails}/{TEST_SHOTS} fails "
          f"(LER={ler:.5f})")
    if not (0.005 <= ler <= 0.03):
        print("  WARNING: this is outside the ~1.3-1.5% range expected at p=0.005, "
              "distance 5 -- double check BASIS/P/DISTANCE before proceeding.")
    else:
        print("  LER is in the expected range.")

    print("\nDone. Next steps:")
    print("  1. python decoder_triage_pipeline.py")
    print("  2. python app1.py")


if __name__ == "__main__":
    main()