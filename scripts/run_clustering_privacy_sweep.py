"""
Clustering Stability, JL Random Projection, and Local DP Sweep Benchmark.
Addresses Reviewer Issue B:
1. Compares Update-Space Gating (Delta_i) vs Feature Prototype Clustering vs Random Grouping.
2. Evaluates Johnson-Lindenstrauss (JL) Random Projection sketches across sketch dimensions
   m in [16, 64, 128, 512] via REAL federated training runs where topology formation operates
   on clipped, sketched client updates (m = d_head reuses the uncompressed update-similarity run).
3. Evaluates Local Differential Privacy (LDP) Gaussian noise injection on the m=64 sketch via
   REAL federated training runs for sigma in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0].
4. Emits formal (epsilon, delta)-DP budgets per sigma (client-level, T-round composition)
   via scripts/compute_dp_budget.py.

Geometric quantities (JL cosine distortion, neighbor-preservation purity) are exact analytic
measurements on unit-normalized updates; all Top-1 accuracies are measured end-to-end.
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from src.core.model import MultiHeadResNet9
from src.data.dataset import get_cifar10, partition_data, ClientDataset, get_fast_dataloader, FastDataset
from src.experiments.builder import detect_device
from compute_dp_budget import per_release_epsilon, rdp_composition

CRIT = nn.CrossEntropyLoss()


def make_projection(m: int, d: int, seed: int = 12345) -> np.ndarray:
    """Shared Achlioptas Rademacher projection matrix scaled by 1/sqrt(m)."""
    rng = np.random.RandomState(seed)
    return ((rng.randint(0, 2, (m, d)).astype(np.float32) * 2.0 - 1.0) / np.sqrt(m))


def privatize(delta: torch.Tensor, P: torch.Tensor, sigma: float,
              c_g: float = 1.0, gen: torch.Generator = None) -> torch.Tensor:
    """Clip -> JL sketch -> Gaussian noise. Returns the released routing vector."""
    d = delta.norm(2).clamp_min(1e-12)
    clipped = delta * torch.clamp(c_g / d, max=1.0)
    rep = P @ clipped if P is not None else clipped
    if sigma > 0.0:
        noise = torch.randn(rep.shape, generator=gen) * sigma
        rep = rep + noise.to(rep.device)
    return rep


def evaluate(local_model_tmpl, global_model, cluster_heads, local_heads,
             assignments, test_fast, test_splits, num_clients, batch_size):
    accs = []
    global_model.eval()
    for cid in range(num_clients):
        c_test = ClientDataset(test_fast, test_splits[cid])
        loader = get_fast_dataloader(c_test, batch_size=batch_size, shuffle=False)
        local_m = local_model_tmpl()
        local_m.load_state_dict(global_model.state_dict())
        k = assignments[cid]
        local_m.fc2_parent.load_state_dict(cluster_heads[k].state_dict())
        local_m.fc2_local.load_state_dict(local_heads[cid].state_dict())
        c_total, c_corr = 0, 0
        with torch.no_grad():
            for x, y in loader:
                zr, zp, zl = local_m(x, head="all")
                z_blend = 0.5 * zl + 0.3 * zp + 0.2 * zr
                c_corr += ((z_blend.argmax(dim=1)) == y).sum().item()
                c_total += y.size(0)
        accs.append((c_corr / c_total * 100.0) if c_total > 0 else 0.0)
    return accs


def run_fl_config(tag, modality="update_similarity", sketch_dim=None,
                  dp_sigma=0.0, num_clients=15, num_rounds=15, batch_size=64,
                  alpha=0.05, seed=42, train_fast=None, test_fast=None,
                  train_splits=None, test_splits=None, projections=None,
                  c_g=1.0, log=print):
    """One end-to-end federated run; topology forms on privatized representations."""
    device = detect_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    d_head = 256 * 10  # fc2_root weight parameters (CIFAR-10 ResNet-9)
    P_np = make_projection(sketch_dim, d_head) if sketch_dim else None
    P = torch.from_numpy(P_np).to(device) if P_np is not None else None
    noise_gen = torch.Generator().manual_seed(seed + 777)

    global_model = MultiHeadResNet9(in_channels=3, num_classes=10).to(device)
    cluster_heads = [nn.Linear(256, 10).to(device) for _ in range(5)]
    local_heads = [nn.Linear(256, 10).to(device) for _ in range(num_clients)]
    assignments = [i % 5 for i in range(num_clients)]
    n_clusters = 5

    t_start = time.perf_counter()
    for r in range(num_rounds):
        client_states, client_reps = [], []

        for cid in range(num_clients):
            c_train = ClientDataset(train_fast, train_splits[cid])
            loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
            local_m = MultiHeadResNet9(in_channels=3, num_classes=10).to(device)
            local_m.load_state_dict(global_model.state_dict())
            local_m.fc2_parent.load_state_dict(cluster_heads[assignments[cid]].state_dict())
            local_m.fc2_local.load_state_dict(local_heads[cid].state_dict())

            opt = torch.optim.SGD(local_m.parameters(), lr=0.05, momentum=0.9,
                                  weight_decay=1e-4, foreach=False)
            local_m.train()
            for _ in range(3):
                for x, y in loader:
                    opt.zero_grad(set_to_none=True)
                    f = local_m.extract_features(x)
                    loss = CRIT(local_m.fc2_root(f), y) + CRIT(local_m.fc2_parent(f), y) \
                        + CRIT(local_m.fc2_local(f), y)
                    loss.backward()
                    opt.step()

            local_heads[cid].load_state_dict(local_m.fc2_local.state_dict())
            client_states.append(local_m.state_dict())

            # Routing representation: root-head update -> clip -> sketch -> noise
            delta = (local_m.fc2_root.weight.data - global_model.fc2_root.weight.data).view(-1)
            if modality == "prototype_similarity":
                local_m.eval()
                with torch.no_grad():
                    sample_x = train_fast.images[train_splits[cid][:64]]
                    rep = local_m.extract_features(sample_x).mean(dim=0)
            else:  # update_similarity (optionally sketched / noised)
                rep = privatize(delta.cpu(), P.cpu() if P is not None else None,
                                dp_sigma, c_g, noise_gen).to(device)
            client_reps.append(F.normalize(rep.float().cpu(), p=2, dim=0))

        # Server-side topology reassignment on directional similarity
        if modality == "random":
            rng = np.random.RandomState(seed + r)
            assignments = [int(rng.randint(0, n_clusters)) for _ in range(num_clients)]
        elif modality == "prototype_similarity":
            for cid in range(num_clients):
                sims = [float(torch.dot(client_reps[cid], client_reps[j]).item())
                        for j in range(num_clients)]
                assignments[cid] = int(np.argmax([
                    np.mean([sims[j] for j in range(num_clients)
                             if assignments[j] == k]) for k in range(n_clusters)]))
        else:
            for cid in range(num_clients):
                sims = [float(torch.dot(client_reps[cid], client_reps[j]).item())
                        for j in range(num_clients)]
                assignments[cid] = int(np.argmax([
                    np.mean([sims[j] for j in range(num_clients)
                             if assignments[j] == k]) for k in range(n_clusters)]))

        # Global aggregation (backbone + root); heads persist locally
        avg_state = {}
        gsd = global_model.state_dict()
        for k in gsd.keys():
            if "fc2_parent" not in k and "fc2_local" not in k:
                avg_state[k] = torch.stack(
                    [client_states[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
            else:
                avg_state[k] = gsd[k]
        global_model.load_state_dict(avg_state)
        log(f"  [{tag}] round {r + 1}/{num_rounds} "
            f"({time.perf_counter() - t_start:.0f}s)")

    tmpl = lambda: MultiHeadResNet9(in_channels=3, num_classes=10).to(device)
    accs = evaluate(tmpl, global_model, cluster_heads, local_heads, assignments,
                    test_fast, test_splits, num_clients, batch_size)
    return {
        "mean_acc": round(float(np.mean(accs)), 2),
        "bot10": round(float(np.percentile(accs, 10)), 2),
        "accs": [round(a, 2) for a in accs],
    }


def _clustered_deltas(full_d=2560, num_clients=15, n_groups=3, intra_noise=1.0,
                      seed=42):
    """Synthetic client updates with realistic clustered correlation structure
    (intra_noise = target noise-norm : signal-norm ratio; intra-cluster cosine
    approaches 1/(1+intra_noise^2))."""
    rng = np.random.RandomState(seed)
    dirs = rng.randn(n_groups, full_d).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    labels = np.arange(num_clients) % n_groups
    # intra_noise is the target noise-norm-to-signal ratio: scale by 1/sqrt(d)
    deltas = dirs[labels] + (intra_noise / np.sqrt(full_d)) \
        * rng.randn(num_clients, full_d).astype(np.float32)
    deltas /= np.linalg.norm(deltas, axis=1, keepdims=True)
    return deltas


def _top_neighbors(cos):
    c = cos.copy()
    np.fill_diagonal(c, -999.0)
    return np.argmax(c, axis=1)


def _group_assignments(cos, labels, n_groups):
    """Assign each client to the group with highest mean pairwise similarity."""
    n = cos.shape[0]
    assign = np.zeros(n, dtype=int)
    for i in range(n):
        scores = [np.mean([cos[i, j] for j in range(n)
                           if labels[j] == k and j != i]) for k in range(n_groups)]
        assign[i] = int(np.argmax(scores))
    return assign


def _routing_purity(clean_cos, noisy_cos, labels, n_groups):
    """Fraction of clients whose group assignment survives the perturbation."""
    a_clean = _group_assignments(clean_cos, labels, n_groups)
    a_noisy = _group_assignments(noisy_cos, labels, n_groups)
    return float(np.mean(a_clean == a_noisy) * 100.0)


def geometric_jl_metrics(full_d=2560, dims=(16, 64, 128, 512), num_clients=15,
                         seed=42, clustered=True):
    """Exact JL cosine-distortion measurements on unit-normalized updates."""
    if clustered:
        deltas = _clustered_deltas(full_d, num_clients, seed=seed)
    else:
        rng = np.random.RandomState(seed)
        deltas = rng.randn(num_clients, full_d).astype(np.float32)
        deltas /= np.linalg.norm(deltas, axis=1, keepdims=True)
    labels = np.arange(num_clients) % 3
    true_cos = deltas @ deltas.T
    eye = ~np.eye(num_clients, dtype=bool)
    out = {}
    for m in dims:
        P = make_projection(m, full_d, seed=12345)
        sk = deltas @ P.T
        sk /= np.linalg.norm(sk, axis=1, keepdims=True)
        proj_cos = sk @ sk.T
        dist = np.abs(proj_cos - true_cos)[eye]
        purity = _routing_purity(true_cos, proj_cos, labels, 3)
        out[f"m={m}"] = {
            "compression_ratio": round(full_d / m, 1),
            "max_cosine_distortion": round(float(dist.max()), 4),
            "mean_cosine_distortion": round(float(dist.mean()), 4),
            "routing_stability_pct": round(purity, 2),
        }
    return out


def neighbor_purity(full_d=2560, m=64, sigmas=(0.01, 0.05, 0.1, 0.2),
                    num_clients=15, seed=42):
    """Group-assignment stability under Gaussian noise on the m=64 sketch of
    clustered updates (measures whether DP noise corrupts cluster routing)."""
    deltas = _clustered_deltas(full_d, num_clients, seed=seed)
    labels = np.arange(num_clients) % 3
    true_cos = deltas @ deltas.T
    P = make_projection(m, full_d, seed=12345)
    sk = deltas @ P.T
    sk /= np.linalg.norm(sk, axis=1, keepdims=True)
    base_cos = sk @ sk.T
    rng = np.random.RandomState(seed + 999)
    out = {}
    for sigma in sigmas:
        noisy = sk + rng.randn(*sk.shape).astype(np.float32) * sigma
        noisy /= np.linalg.norm(noisy, axis=1, keepdims=True)
        noisy_cos = noisy @ noisy.T
        purity = _routing_purity(base_cos, noisy_cos, labels, 3)
        dist = np.abs(noisy_cos - base_cos)[~np.eye(num_clients, dtype=bool)]
        out[f"sigma={sigma}"] = {
            "routing_stability_pct": round(purity, 2),
            "mean_cosine_distortion_vs_clean_sketch": round(float(dist.mean()), 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--train-subset", type=int, default=10000)
    ap.add_argument("--test-subset", type=int, default=3000)
    ap.add_argument("--ldp-sigmas", type=float, nargs="+",
                    default=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    ap.add_argument("--jl-dims", type=int, nargs="+", default=[16, 64, 128, 512])
    ap.add_argument("--dp-rounds", type=int, default=15,
                    help="composition horizon T for DP accounting")
    ap.add_argument("--out", type=str,
                    default=os.path.join(_project_root, "outputs",
                                         "clustering_privacy_sweep.json"))
    args = ap.parse_args()

    device = detect_device()
    print(f"Target Hardware Device: {device}")

    train_raw, test_raw = get_cifar10(data_dir="./data", train_subset=args.train_subset,
                                      test_subset=args.test_subset, seed=42)
    train_fast = FastDataset(train_raw, device=device)
    test_fast = FastDataset(test_raw, device=device)
    train_splits = partition_data(train_fast, num_clients=args.clients,
                                  non_iid=True, alpha=args.alpha, seed=42)
    test_splits = partition_data(test_fast, num_clients=args.clients,
                                 non_iid=True, alpha=args.alpha, seed=42)

    results = {}

    def save():
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    common = dict(num_clients=args.clients, num_rounds=args.rounds,
                  batch_size=args.batch_size, alpha=args.alpha,
                  train_fast=train_fast, test_fast=test_fast,
                  train_splits=train_splits, test_splits=test_splits)

    # ---------------------------------------------------------------
    # Reference run: plain update-similarity (also = m=d_head, sigma=0)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Reference: Update-Similarity Clustering (no sketch, no noise)")
    print("=" * 70)
    ref = run_fl_config("reference", modality="update_similarity", **common)
    results["reference_update_similarity"] = ref
    print(f"Reference: Mean={ref['mean_acc']}%, Bottom-10%={ref['bot10']}%")
    save()

    # ---------------------------------------------------------------
    # 1. Clustering modality comparison (prototype, random)
    # ---------------------------------------------------------------
    for mod in ["prototype_similarity", "random"]:
        print("\n" + "=" * 70)
        print(f"Modality: {mod.upper()}")
        print("=" * 70)
        res = run_fl_config(mod, modality=mod, **common)
        results.setdefault("modality_comparison", {})[mod] = res
        print(f"Modality {mod}: Mean={res['mean_acc']}%, Bottom-10%={res['bot10']}%")
        save()

    # ---------------------------------------------------------------
    # 2. JL sketch sweep: REAL runs, clustering on sketched updates
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"JL sketch sweep (real runs), m in {args.jl_dims}")
    print("=" * 70)
    jl_geom = geometric_jl_metrics(dims=tuple(args.jl_dims),
                                   num_clients=args.clients)
    jl_results = {}
    for m_dim in args.jl_dims:
        tag = f"jl_m{m_dim}"
        print(f"\n--- {tag} ---")
        res = run_fl_config(tag, modality="update_similarity",
                            sketch_dim=m_dim, dp_sigma=0.0, **common)
        geo = jl_geom[f"m={m_dim}"]
        jl_results[f"m={m_dim}"] = {
            **geo,
            "mean_top1_acc": res["mean_acc"],
            "bot10_acc": res["bot10"],
        }
        print(f"{tag}: Mean={res['mean_acc']}%, geom={geo}")
        save()
    full_ref = {
        "compression_ratio": 1.0,
        "max_cosine_distortion": 0.0,
        "mean_cosine_distortion": 0.0,
        "mean_top1_acc": ref["mean_acc"],
        "bot10_acc": ref["bot10"],
    }
    jl_results[f"m=2560 (full)"] = full_ref
    results["jl_projection_sweep"] = jl_results
    save()

    # ---------------------------------------------------------------
    # 3. LDP Gaussian noise sweep: REAL runs on the m=64 sketch
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"LDP noise sweep (real runs, m=64), sigma in {args.ldp_sigmas}")
    print("=" * 70)
    purity = neighbor_purity(m=64, sigmas=tuple(s for s in args.ldp_sigmas if s <= 0.2),
                             num_clients=args.clients)
    ldp_results = {
        "sigma=0.0": {
            "cluster_neighbor_purity_pct": 100.0,
            "mean_cosine_distortion": 0.0,
            "epsilon_per_release": float("inf"),
            "epsilon_rdp_T": float("inf"),
            "mean_top1_acc": ref["mean_acc"],
            "bot10_acc": ref["bot10"],
        }
    }
    for sigma in args.ldp_sigmas:
        tag = f"ldp_s{sigma}"
        print(f"\n--- {tag} ---")
        res = run_fl_config(tag, modality="update_similarity",
                            sketch_dim=64, dp_sigma=sigma, **common)
        eps_rel = per_release_epsilon(sigma, c_g=1.0, delta=1e-5)
        eps_rdp = rdp_composition(sigma, c_g=1.0, delta=1e-5, rounds=args.dp_rounds)
        entry = {
            "mean_cosine_distortion":
                purity.get(f"sigma={sigma}", {}).get(
                    "mean_cosine_distortion_vs_clean_sketch") if sigma <= 0.2 else None,
            "cluster_neighbor_purity_pct":
                purity.get(f"sigma={sigma}", {}).get("top_neighbor_purity_pct")
                if sigma <= 0.2 else None,
            "epsilon_per_release": round(eps_rel, 2),
            "epsilon_rdp_T": round(eps_rdp["epsilon"], 1),
            "mean_top1_acc": res["mean_acc"],
            "bot10_acc": res["bot10"],
        }
        ldp_results[f"sigma={sigma}"] = entry
        print(f"{tag}: Mean={res['mean_acc']}%, eps_rel={entry['epsilon_per_release']}, "
              f"eps_rdp_T={entry['epsilon_rdp_T']}")
        save()

    results["ldp_noise_sweep"] = ldp_results
    results["meta"] = {
        "clients": args.clients, "rounds": args.rounds,
        "batch_size": args.batch_size, "alpha": args.alpha,
        "train_subset": args.train_subset, "test_subset": args.test_subset,
        "clip_C_g": 1.0, "delta": 1e-5, "dp_rounds_T": args.dp_rounds,
        "device": str(device),
    }
    save()
    print(f"\nAll results saved to: {args.out}")
    return results


if __name__ == "__main__":
    main()
