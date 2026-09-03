"""
decoder_triage_pipeline.py

Trains a spatiotemporal GNN to predict, per shot, whether MWPM is likely to
mis-decode a distance-5 rotated surface code memory experiment. Evaluates the
GNN's ranking against two baselines (raw syndrome weight, and a K=16 MWPM
ensemble-disagreement score) as a triage/escalation policy: the top-X% riskiest
shots (by each method's score) are re-decoded with BP+OSD instead of MWPM.

Outputs (all under outputs/):
  triage_results/roc_curve.png     -- triage classifiers' ability to flag
                                       shots MWPM will get wrong
  triage_results/ler_tradeoff.png  -- effective logical error rate vs.
                                       escalation rate, for every strategy
  triage_models/triage_gnn_state.pt -- trained model weights
  triage_models/scores_test.npz     -- real test-set scores/labels/latencies
                                        and full escalation sweep, for use by
                                        a downstream application (see app.py)
"""

import os
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool
import pymatching
import stim
from stimbposd import BPOSD
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASIS = "Z"
P = 0.005  
DISTANCE = 5
ROUNDS = 5

DATA_ROOT = os.path.join("outputs", "edge_reweight_data")
OUTPUT_ROOT = os.path.join("outputs", "triage_results")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Where the trained model + real (measured, not hardcoded) scores/latencies get
# saved so the live dashboard can load actual pipeline output instead of
# fabricated numbers.
MODEL_ROOT = os.path.join("outputs", "triage_models")
os.makedirs(MODEL_ROOT, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_ROOT, "triage_gnn_state.pt")
SCORES_PATH = os.path.join(MODEL_ROOT, "scores_test.npz")
LATENCY_N_TRIALS = 200  # shots used to time each decoder for a stable average

STRUCTURE_PATH = os.path.join(DATA_ROOT, f"dem_structure_{BASIS}_p{P:.4f}.json")
TRAIN_PATH = os.path.join(DATA_ROOT, f"train_{BASIS}_p{P:.4f}.npz")
TEST_PATH = os.path.join(DATA_ROOT, f"test_{BASIS}_p{P:.4f}.npz")

SPATIAL_STEP_RADIUS = 1
HIDDEN_DIM = 64
NUM_GAT_LAYERS = 3
ATTENTION_HEADS = 4
DROPOUT = 0.2

TRAIN_SUBSET_SHOTS = 100_000
TEST_SUBSET_SHOTS = 50_000
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-3

MAX_BP_ITERS = 20
N_PERTURBATIONS = 16

# ----------------------------------------------------------------------
# Professional Plotting Setup
# ----------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.5,
    'grid.linestyle': '--'
})

# ----------------------------------------------------------------------
# Fast Static Graph Construction
# ----------------------------------------------------------------------
def build_static_node_graph(coords: np.ndarray, radius: int = SPATIAL_STEP_RADIUS):
    n = coords.shape[0]
    x_steps, y_steps, t_steps = coords[:, 0] / 2.0, coords[:, 1] / 2.0, coords[:, 2] / 1.0
    dx_m = np.abs(x_steps[:, None] - x_steps[None, :])
    dy_m = np.abs(y_steps[:, None] - y_steps[None, :])
    dt_m = np.abs(t_steps[:, None] - t_steps[None, :])
    neighbor_mask = (dx_m <= radius) & (dy_m <= radius) & (dt_m <= radius)
    np.fill_diagonal(neighbor_mask, False)
    src, dst = np.nonzero(neighbor_mask)
    
    has_neighbor = np.zeros(n, dtype=bool)
    if len(src) > 0: has_neighbor[src] = True
    isolated = np.nonzero(~has_neighbor)[0]
    if len(isolated) > 0:
        src = np.concatenate([src, isolated])
        dst = np.concatenate([dst, isolated])
        
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    diffs = coords[src] - coords[dst]
    inv_dists = (1.0 / (np.sqrt(np.sum(diffs ** 2, axis=1)) + 1e-6)).reshape(-1, 1)
    edge_attr = torch.tensor(np.concatenate([diffs, inv_dists], axis=1), dtype=torch.float32)
    return edge_index, edge_attr

def make_node_features(coords: np.ndarray, basis_flag: float, fired_mask: np.ndarray) -> np.ndarray:
    x_steps, y_steps = coords[:, 0] / 2.0, coords[:, 1] / 2.0
    parity = np.where(((x_steps + y_steps).round().astype(int) % 2) == 0, 1.0, -1.0).astype(np.float32)
    is_first_round = (coords[:, 2] == 0).astype(np.float32)
    return np.stack([parity, is_first_round, np.full((coords.shape[0],), basis_flag, dtype=np.float32), fired_mask.astype(np.float32)], axis=1).astype(np.float32)

# ----------------------------------------------------------------------
# Triage Classifier Architecture
# ----------------------------------------------------------------------
class TriageGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_embed_dim = HIDDEN_DIM * ATTENTION_HEADS
        self.proj = nn.Linear(4, self.node_embed_dim)
        self.convs = nn.ModuleList([GATConv(self.node_embed_dim, HIDDEN_DIM, heads=ATTENTION_HEADS, edge_dim=4, dropout=DROPOUT, concat=True) for _ in range(NUM_GAT_LAYERS)])
        self.classifier = nn.Sequential(nn.Linear(self.node_embed_dim, HIDDEN_DIM), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN_DIM, 1))

    def forward(self, x, edge_index, edge_attr, batch_idx):
        x = self.proj(x)
        for conv in self.convs:
            x = x + F.elu(conv(x, edge_index, edge_attr=edge_attr))
        return self.classifier(global_mean_pool(x, batch_idx)).squeeze(-1)

# ----------------------------------------------------------------------
# Memory-Safe Dynamic Data Generator
# ----------------------------------------------------------------------
def batch_generator(det_array, fail_labels, coords, basis_flag, static_edge_index, static_edge_attr, batch_size, shuffle=True):
    indices = np.arange(det_array.shape[0])
    if shuffle: np.random.shuffle(indices)
    for start_idx in range(0, len(indices), batch_size):
        batch_idx = indices[start_idx:start_idx + batch_size]
        graphs = [Data(x=torch.tensor(make_node_features(coords, basis_flag, det_array[i]), dtype=torch.float32), edge_index=static_edge_index, edge_attr=static_edge_attr) for i in batch_idx]
        yield Batch.from_data_list(graphs).to(DEVICE), torch.tensor([fail_labels[i] for i in batch_idx], dtype=torch.float32).to(DEVICE)

# ----------------------------------------------------------------------
# Main Orchestration
# ----------------------------------------------------------------------
def main():
    # Set seeds for reproducible, highly optimized convergence
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    print(f"=== QUANTUM OS ROUTING ENGINE: End-to-End Evaluation ===")
    
    with open(STRUCTURE_PATH) as f: s = json.load(f)
    coords, basis_flag = np.array(s["coords"], dtype=np.float32), s["basis_flag"]
    static_edge_index, static_edge_attr = build_static_node_graph(coords)
    
    print("\n1. Loading Simulated Dataset...")
    train_data, test_data = np.load(TRAIN_PATH), np.load(TEST_PATH)
    n_train = min(TRAIN_SUBSET_SHOTS, train_data["detection_events"].shape[0])
    n_test = min(TEST_SUBSET_SHOTS, test_data["detection_events"].shape[0])
    
    train_det, train_obs = train_data["detection_events"][:n_train], train_data["observable_flips"][:n_train]
    test_det, test_obs = test_data["detection_events"][:n_test], test_data["observable_flips"][:n_test]
    train_data.close(); test_data.close()

    print("\n2. Generating Baseline Decodes (MWPM & BP+OSD)...")
    dem = stim.Circuit.generated(f"surface_code:rotated_memory_{BASIS.lower()}", distance=DISTANCE, rounds=ROUNDS, after_clifford_depolarization=P, after_reset_flip_probability=P, before_measure_flip_probability=P, before_round_data_depolarization=P).detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    
    train_fail_labels = np.any(matcher.decode_batch(train_det) != train_obs, axis=1).astype(np.float32)
    test_fail_labels = np.any(matcher.decode_batch(test_det) != test_obs, axis=1).astype(np.float32)
    mwpm_ler_test = test_fail_labels.mean()
    print(f"  MWPM LER -> Train: {train_fail_labels.mean():.5f} | Test: {mwpm_ler_test:.5f}")

    bposd = BPOSD(dem, max_bp_iters=MAX_BP_ITERS)
    test_bposd_fails = np.any(bposd.decode_batch(test_det) != test_obs, axis=1).astype(np.float32)
    bposd_ler_test = test_bposd_fails.mean()
    print(f"  BP+OSD LER -> Test: {bposd_ler_test:.5f}")

    print("\n3. Computing Classical Baselines (Fair Environment)...")
    sw_scores = test_det.sum(axis=1).astype(np.float32)
    
    print("  Computing K=16 Classical Ensemble (Takes ~10s)...")
    t0 = time.time()
    edges_list, base_w = list(matcher.edges()), np.array([e[2]["weight"] for e in matcher.edges()], dtype=np.float64)
    rng, disagree = np.random.default_rng(42), np.zeros(n_test, dtype=np.int32)
    test_mwpm_pred = matcher.decode_batch(test_det)[:, 0].astype(bool)
    
    for _ in range(N_PERTURBATIONS):
        jw = base_w * np.exp(rng.normal(0.0, 0.2, size=base_w.shape[0]))
        mk = pymatching.Matching()
        for (u, v, data), w in zip(edges_list, jw):
            if v is None: mk.add_boundary_edge(u, weight=float(w), fault_ids=data.get("fault_ids", set()))
            else: mk.add_edge(u, v, weight=float(w), fault_ids=data.get("fault_ids", set()))
        disagree += (mk.decode_batch(test_det.astype(np.uint8))[:, 0].astype(bool) != test_mwpm_pred).astype(np.int32)
        
    ens_scores = disagree.astype(np.float32) / N_PERTURBATIONS
    print(f"  Ensemble computed in {time.time()-t0:.1f}s")

    print("\n4. Training Proposed Model: Spatiotemporal GNN...")
    model = TriageGNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    pos_weight = torch.tensor((n_train - train_fail_labels.sum()) / max(train_fail_labels.sum(), 1.0), dtype=torch.float32).to(DEVICE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n_seen = 0.0, 0
        t0 = time.time()
        for batch, labels in batch_generator(train_det, train_fail_labels, coords, basis_flag, static_edge_index, static_edge_attr, BATCH_SIZE, shuffle=True):
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch), labels, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            n_seen += batch.num_graphs
        print(f"  Epoch {epoch:02d} | BCE Loss: {total_loss/n_seen:.4f} | Time: {time.time() - t0:.1f}s")

    print("\n5. Extracting GNN Predictions...")
    model.eval()
    gnn_scores = []
    with torch.no_grad():
        for batch, _ in batch_generator(test_det, test_fail_labels, coords, basis_flag, static_edge_index, static_edge_attr, BATCH_SIZE, shuffle=False):
            gnn_scores.extend(torch.sigmoid(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)).cpu().numpy())
    gnn_scores = np.array(gnn_scores)

    print("\n6. Measuring Real Per-Shot Decoder Latency...")
    # NOTE ON A FIX: an earlier version of this timed the GNN with batch_size=1,
    # which mostly measures Python's cost of building a torch_geometric.Data
    # object per shot, not the model itself -- that gave a nonsense ~89ms/shot
    # figure. Below, the batch is built once, outside the timer, so only the
    # actual forward pass is measured.
    trial_det = test_det[:LATENCY_N_TRIALS]

    t0 = time.time()
    matcher.decode_batch(trial_det)
    latency_mwpm_us = (time.time() - t0) / LATENCY_N_TRIALS * 1e6

    trial_batch, _ = next(batch_generator(trial_det, np.zeros(LATENCY_N_TRIALS), coords, basis_flag,
                                           static_edge_index, static_edge_attr,
                                           batch_size=LATENCY_N_TRIALS, shuffle=False))
    model.eval()
    LATENCY_REPEATS = 20
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(LATENCY_REPEATS):
            model(trial_batch.x, trial_batch.edge_index, trial_batch.edge_attr, trial_batch.batch)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    latency_gnn_us = (time.time() - t0) / (LATENCY_REPEATS * LATENCY_N_TRIALS) * 1e6
    # every shot gets the MWPM pass regardless; the GNN forward pass rides on top of it
    latency_gnn_us += latency_mwpm_us

    t0 = time.time()
    bposd.decode_batch(trial_det)
    latency_bposd_us = (time.time() - t0) / LATENCY_N_TRIALS * 1e6

    print(f"  MWPM:   {latency_mwpm_us:.2f} us/shot")
    print(f"  GNN:    {latency_gnn_us:.2f} us/shot (MWPM + routing forward pass)")
    print(f"  BP+OSD: {latency_bposd_us:.2f} us/shot")
    print(f"  Classical Ensemble (K={N_PERTURBATIONS}): approx {N_PERTURBATIONS} x MWPM = "
          f"{N_PERTURBATIONS * latency_mwpm_us:.2f} us/shot (K independent MWPM decodes)")
    latency_ens_us = N_PERTURBATIONS * latency_mwpm_us

    print("\n7. Computing Escalation-Rate Tradeoff Curves...")
    thresholds = np.linspace(0.0, 0.30, 100)  # 0% to 30% escalation

    sw_idx = np.argsort(-sw_scores)
    ens_idx = np.argsort(-ens_scores)
    gnn_idx = np.argsort(-gnn_scores)
    rand_idx = np.random.permutation(n_test)

    def get_curve(sorted_indices):
        ler_vals = []
        for t in thresholds:
            n_flag = int(n_test * t)
            fails = test_bposd_fails[sorted_indices[:n_flag]].sum() + test_fail_labels[sorted_indices[n_flag:]].sum()
            ler_vals.append(fails / n_test)
        return np.array(ler_vals)

    ler_sw = get_curve(sw_idx)
    ler_ens = get_curve(ens_idx)
    ler_gnn = get_curve(gnn_idx)
    ler_rand = get_curve(rand_idx)

    print("\n8. Saving Model + Real Test Scores (for downstream use, e.g. an application script)...")
    torch.save(model.state_dict(), MODEL_PATH)
    np.savez(
        SCORES_PATH,
        test_fail_labels=test_fail_labels,        # ground truth: did MWPM fail this shot?
        test_bposd_fail_labels=test_bposd_fails,   # ground truth: did BP+OSD fail this shot?
        sw_scores=sw_scores,                       # syndrome-weight baseline risk score
        ens_scores=ens_scores,                     # classical ensemble risk score
        gnn_scores=gnn_scores,                     # our GNN's risk score
        mwpm_ler_test=mwpm_ler_test,
        bposd_ler_test=bposd_ler_test,
        latency_mwpm_us=latency_mwpm_us,
        latency_gnn_us=latency_gnn_us,
        latency_bposd_us=latency_bposd_us,
        latency_ens_us=latency_ens_us,
        # full escalation-rate sweep for every triage strategy, so a downstream
        # application can pick an operating point (e.g. 10%) and get each
        # strategy's real effective LER at that point instead of guessing
        escalation_thresholds=thresholds,
        ler_curve_syndrome_weight=ler_sw,
        ler_curve_ensemble=ler_ens,
        ler_curve_gnn=ler_gnn,
        ler_curve_random=ler_rand,
    )
    print(f"  Model saved to:  {MODEL_PATH}")
    print(f"  Scores saved to: {SCORES_PATH}")

    print("\n9. Generating Professional Academic Plots...")
    
    # ----------------------------------------------------------------------
    # PLOT 1: ROC Curve
    # ----------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    
    fpr_sw, tpr_sw, _ = roc_curve(test_fail_labels, sw_scores)
    auc_sw = roc_auc_score(test_fail_labels, sw_scores)
    
    fpr_ens, tpr_ens, _ = roc_curve(test_fail_labels, ens_scores)
    auc_ens = roc_auc_score(test_fail_labels, ens_scores)
    
    fpr_gnn, tpr_gnn, _ = roc_curve(test_fail_labels, gnn_scores)
    auc_gnn = roc_auc_score(test_fail_labels, gnn_scores)
    
    plt.plot(fpr_sw, tpr_sw, color='gray', linestyle='-.', lw=2.5, label=f'Syndrome Weight Baseline (AUC = {auc_sw:.3f})')
    plt.plot(fpr_ens, tpr_ens, color='#1f77b4', linestyle='--', lw=2.5, label=f'Classical Ensemble Upper Bound (AUC = {auc_ens:.3f})')
    plt.plot(fpr_gnn, tpr_gnn, color='#d62728', linestyle='-', lw=2.5, label=f'Spatiotemporal GNN [Ours] (AUC = {auc_gnn:.3f})')
    plt.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle=':')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Escalating Normal Shots)')
    plt.ylabel('True Positive Rate (Catching MWPM Failures)')
    plt.title('ROC Curve: Decoder Triage Classification')
    plt.legend(loc="lower right")
    
    roc_path = os.path.join(OUTPUT_ROOT, 'roc_curve.png')
    plt.savefig(roc_path, bbox_inches='tight')
    plt.close()
    
    # ----------------------------------------------------------------------
    # PLOT 2: Tradeoff Curve (Escalation vs LER) -- reuses the curves already
    # computed and saved above, no recomputation
    # ----------------------------------------------------------------------
    plt.figure(figsize=(8, 6))

    plt.plot(thresholds * 100, ler_rand, color='black', linestyle=':', lw=2.5, label='Random Escalation')
    plt.plot(thresholds * 100, ler_sw, color='gray', linestyle='-.', lw=2.5, label='Syndrome Weight Triage')
    plt.plot(thresholds * 100, ler_ens, color='#1f77b4', linestyle='--', lw=2.5, label='Classical Ensemble Triage')
    plt.plot(thresholds * 100, ler_gnn, color='#d62728', linestyle='-', lw=2.5, label='GNN Triage [Ours]')
    
    plt.axhline(y=mwpm_ler_test, color='black', alpha=0.5, linestyle='-', label=f'MWPM Baseline / Error Ceiling ({mwpm_ler_test:.4f})')
    plt.axhline(y=bposd_ler_test, color='green', alpha=0.5, linestyle='-', label=f'BP+OSD Limit / Error Floor ({bposd_ler_test:.4f})')
    
    plt.xlim([0, 30])
    plt.ylim([bposd_ler_test * 0.9, mwpm_ler_test * 1.05])
    plt.xlabel('Escalation Rate (% of shots routed to heavy decoder)')
    plt.ylabel('Effective Logical Error Rate (LER)')
    plt.title('Latency vs. Accuracy Tradeoff in Adaptive Decoding')
    plt.legend(loc="upper right")
    
    tradeoff_path = os.path.join(OUTPUT_ROOT, 'ler_tradeoff.png')
    plt.savefig(tradeoff_path, bbox_inches='tight')
    plt.close()

    print("\n" + "="*80)
    print(" DONE")
    print("="*80)
    print(f" ROC Curve:       {roc_path}")
    print(f" Tradeoff Curve:  {tradeoff_path}")
    print(f" Model + scores:  {MODEL_PATH}, {SCORES_PATH}")

if __name__ == "__main__": 
    main()
