"""
50-Client Partial-Participation Scalability Benchmark (N=50, Cp=0.2) -- CIFAR-100.

Redesigned protocol (fixes the 0.00% tail-fairness artifact of the first version):
  * Equal local compute for ALL methods (--local-epochs, default 3 passes over local data).
  * Every method is scored under BOTH protocols on the same test split:
        open   : raw 100-way argmax (no knowledge of the client's label support)
        closed : logits masked to the classes observed in the client's TRAIN split
                 (the personalized-FL protocol used for FedHEP's ACLM inference mask)
    and the closed-set mask is applied to *every* method, not only FedHEP.
  * FedHEP trains parent/local heads with Active-Class Logit Masking (ACLM) and routes
    unsampled/stale clients through the Root head (S-AFR) -- as specified in the paper.
  * One (seed, regime) per process, checkpointed per method, so a crash/hang never loses
    finished work and never blocks other jobs.

Usage (one job):
    python scripts/run_scale_50clients.py --seed 42 --alpha 0.5 --rounds 40
Merge all finished jobs into outputs/scale_50clients_results.json (mean +- std over seeds):
    python scripts/run_scale_50clients.py --merge
"""
import argparse
import copy
import glob
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

from src.core.model import ResNet9
from src.data.dataset import (get_cifar10, get_cifar100, partition_data, ClientDataset,
                              get_fast_dataloader, FastDataset)
from src.experiments.builder import detect_device

OUT_DIR = os.path.join(_project_root, "outputs", "scale50")
REGIMES = {"0.5": "Moderate (alpha=0.5)", "0.1": "Severe (alpha=0.1)"}
METHODS = ["FedAvg", "FedRep", "Ditto", "FedHEP (Ours)"]
LR, MOM, WD = 0.05, 0.9, 1e-4


def _sgd(params):
    return torch.optim.SGD(params, lr=LR, momentum=MOM, weight_decay=WD, foreach=False)


def _avg_states(states):
    return {k: torch.stack([s[k].float() for s in states], dim=0).mean(dim=0) for k in states[0]}


def _summ(accs):
    k = max(1, int(np.ceil(0.1 * len(accs))))
    return {"mean": round(float(np.mean(accs)), 2),
            "bottom10": round(float(np.mean(sorted(accs)[:k])), 2)}


def _masked(logits, mask):
    return logits.masked_fill(~mask, -1e9)


class Ctx:
    """Everything a method needs: data splits, class-support masks, sampling RNG."""

    def __init__(self, args, device):
        self.a, self.device = args, device
        self.nc = 100 if args.dataset == "cifar100" else 10
        loader = get_cifar100 if args.dataset == "cifar100" else get_cifar10
        tr, te = loader(data_dir=args.data_dir, train_subset=args.train_subset,
                        test_subset=args.test_subset, seed=42)
        self.train = FastDataset(tr, device=device)
        self.test = FastDataset(te, device=device)
        alpha = float(args.alpha)
        self.tr_splits = partition_data(self.train, num_clients=args.clients, non_iid=True, alpha=alpha, seed=args.seed)
        self.te_splits = partition_data(self.test, num_clients=args.clients, non_iid=True, alpha=alpha, seed=args.seed)
        # Closed-set support mask per client, derived from TRAIN labels only (no test leakage).
        self.masks = []
        for cid in range(args.clients):
            lab = self.train.labels[self.tr_splits[cid]].cpu().numpy()
            m = np.zeros(self.nc, dtype=bool)
            m[np.unique(lab)] = True
            self.masks.append(torch.tensor(m, device=device))

    def new_rng(self):
        return np.random.RandomState(self.a.seed)

    def loader(self, cid, train=True):
        ds = ClientDataset(self.train if train else self.test, (self.tr_splits if train else self.te_splits)[cid])
        return get_fast_dataloader(ds, batch_size=self.a.batch_size, shuffle=train)

    def evaluate(self, logits_fn):
        """logits_fn(cid, x) -> logits. Returns per-client (open, closed) accuracies in %."""
        open_accs, closed_accs = [], []
        with torch.no_grad():
            for cid in range(self.a.clients):
                tot = co = cc = 0
                for x, y in self.loader(cid, train=False):
                    z = logits_fn(cid, x)
                    co += (z.argmax(1) == y).sum().item()
                    cc += (_masked(z, self.masks[cid]).argmax(1) == y).sum().item()
                    tot += y.size(0)
                open_accs.append(100.0 * co / tot if tot else 0.0)
                closed_accs.append(100.0 * cc / tot if tot else 0.0)
        return open_accs, closed_accs


def _log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------- methods
def run_fedavg(c):
    a, dev = c.a, c.device
    rng, crit = c.new_rng(), nn.CrossEntropyLoss()
    g = ResNet9(in_channels=3, num_classes=c.nc).to(dev)
    for r in range(a.rounds):
        states = []
        for cid in rng.choice(a.clients, a.clients_per_round, replace=False):
            m = copy.deepcopy(g).train()
            opt = _sgd(m.parameters())
            for _ in range(a.local_epochs):
                for x, y in c.loader(cid):
                    opt.zero_grad(set_to_none=True)
                    crit(m(x), y).backward()
                    opt.step()
            states.append(m.state_dict())
        g.load_state_dict(_avg_states(states))
        if (r + 1) % 10 == 0:
            _log(f"    FedAvg round {r + 1}/{a.rounds}")
    g.eval()
    return c.evaluate(lambda cid, x: g(x)), {}


def run_fedrep(c):
    a, dev = c.a, c.device
    rng, crit = c.new_rng(), nn.CrossEntropyLoss()
    g = ResNet9(in_channels=3, num_classes=c.nc).to(dev)
    heads = [nn.Linear(256, c.nc).to(dev) for _ in range(a.clients)]
    for r in range(a.rounds):
        states = []
        for cid in rng.choice(a.clients, a.clients_per_round, replace=False):
            bb = copy.deepcopy(g).train()
            head = heads[cid]
            oh, ob = _sgd(head.parameters()), _sgd(bb.parameters())
            for _ in range(a.local_epochs):       # same total passes as the others: head phase ...
                for x, y in c.loader(cid):
                    oh.zero_grad(set_to_none=True)
                    crit(head(bb.extract_features(x).detach()), y).backward()
                    oh.step()
            for _ in range(a.local_epochs):       # ... then representation phase
                for x, y in c.loader(cid):
                    ob.zero_grad(set_to_none=True)
                    crit(head(bb.extract_features(x)), y).backward()
                    ob.step()
            states.append(bb.state_dict())
        g.load_state_dict(_avg_states(states))
        if (r + 1) % 10 == 0:
            _log(f"    FedRep round {r + 1}/{a.rounds}")
    g.eval()
    [h.eval() for h in heads]
    return c.evaluate(lambda cid, x: heads[cid](g.extract_features(x))), {}


def run_ditto(c, lam=0.1):
    a, dev = c.a, c.device
    rng, crit = c.new_rng(), nn.CrossEntropyLoss()
    g = ResNet9(in_channels=3, num_classes=c.nc).to(dev)
    pers = [ResNet9(in_channels=3, num_classes=c.nc).to(dev) for _ in range(a.clients)]
    for r in range(a.rounds):
        states = []
        for cid in rng.choice(a.clients, a.clients_per_round, replace=False):
            lg = copy.deepcopy(g).train()
            og = _sgd(lg.parameters())
            for _ in range(a.local_epochs):
                for x, y in c.loader(cid):
                    og.zero_grad(set_to_none=True)
                    crit(lg(x), y).backward()
                    og.step()
            states.append(lg.state_dict())
            pm, op = pers[cid].train(), _sgd(pers[cid].parameters())
            wg = torch.nn.utils.parameters_to_vector(lg.parameters()).detach()
            for _ in range(a.local_epochs):
                for x, y in c.loader(cid):
                    op.zero_grad(set_to_none=True)
                    wp = torch.nn.utils.parameters_to_vector(pm.parameters())
                    (crit(pm(x), y) + 0.5 * lam * torch.sum((wp - wg) ** 2)).backward()
                    op.step()
        g.load_state_dict(_avg_states(states))
        if (r + 1) % 10 == 0:
            _log(f"    Ditto round {r + 1}/{a.rounds}")
    [p.eval() for p in pers]
    return c.evaluate(lambda cid, x: pers[cid](x)), {}


def run_fedhep(c, num_clusters=5, beta_c=0.70, stale_tau=4.0):
    a, dev = c.a, c.device
    rng, crit = c.new_rng(), nn.CrossEntropyLoss()
    n, nc, E = a.clients, c.nc, a.local_epochs
    e_r, e_p, e_l = E, max(1, E - 1), max(1, E - 1)    # backbone passes = E (equal to baselines)
    bb = ResNet9(in_channels=3, num_classes=nc).to(dev)
    root = nn.Linear(256, nc).to(dev)
    clus = [nn.Linear(256, nc).to(dev) for _ in range(num_clusters)]
    local = [copy.deepcopy(root) for _ in range(n)]
    cl_of = [i % num_clusters for i in range(n)]       # static round-robin clusters (see README note)
    alphas = [torch.tensor([0.33, 0.33, 0.34], device=dev) for _ in range(n)]
    seen = np.zeros(n, dtype=int)
    last = np.zeros(n, dtype=int)

    priors = []
    for cid in range(n):
        cnt = np.bincount(c.train.labels[c.tr_splits[cid]].cpu().numpy(), minlength=nc)
        p = cnt / (cnt.sum() + 1e-8)
        rs = float(np.clip(-np.sum(p * np.log(p + 1e-12)) / np.log(nc), 0, 1))
        pi_r = rs ** 2
        pi_l = (1 - pi_r) * (1 - rs)
        priors.append(torch.tensor([pi_l, max(0.0, 1 - pi_r - pi_l), pi_r], device=dev))

    for r in range(a.rounds):
        bbs, roots, cupd = [], [], {k: [] for k in range(num_clusters)}
        for cid in rng.choice(n, a.clients_per_round, replace=False):
            seen[cid] += 1
            last[cid] = r
            k = cl_of[cid]
            mask = c.masks[cid]
            lb = copy.deepcopy(bb).train()
            lr_, lp_, ll_ = copy.deepcopy(root).train(), copy.deepcopy(clus[k]).train(), local[cid].train()
            opt = _sgd(list(lb.parameters()) + list(lr_.parameters()) + list(lp_.parameters()) + list(ll_.parameters()))
            loader = c.loader(cid)
            for ep in range(1, max(e_r, e_p, e_l) + 1):
                for x, y in loader:
                    opt.zero_grad(set_to_none=True)
                    f = lb.extract_features(x)
                    loss = 0.0
                    if ep <= e_r: loss = loss + crit(lr_(f), y)                        # Root: global CE
                    if ep <= e_p: loss = loss + crit(_masked(lp_(f), mask), y)         # Parent: ACLM
                    if ep <= e_l: loss = loss + crit(_masked(ll_(f), mask), y)         # Local : ACLM
                    loss.backward()
                    opt.step()
            bbs.append(lb.state_dict()); roots.append(lr_.state_dict()); cupd[k].append(lp_.state_dict())
            lb.eval()
            with torch.no_grad():
                x, y = next(iter(c.loader(cid)))
                f = lb.extract_features(x)
                acc = torch.stack([(_masked(ll_(f), mask).argmax(1) == y).float().mean(),
                                   (_masked(lp_(f), mask).argmax(1) == y).float().mean(),
                                   (lr_(f).argmax(1) == y).float().mean()])
                alphas[cid] = F.softmax(0.7 * acc + 0.3 * priors[cid], dim=0)
        bb.load_state_dict(_avg_states(bbs))
        root.load_state_dict(_avg_states(roots))
        for k in range(num_clusters):                       # asynchronous cluster momentum
            if cupd[k]:
                avg, cur = _avg_states(cupd[k]), clus[k].state_dict()
                clus[k].load_state_dict({kk: beta_c * cur[kk].float() + (1 - beta_c) * avg[kk] for kk in avg})
        if (r + 1) % 10 == 0:
            _log(f"    FedHEP round {r + 1}/{a.rounds}")

    bb.eval(); root.eval(); [h.eval() for h in clus + local]

    def blend(cid, x):
        al = alphas[cid].clone()
        if seen[cid] == 0:                                   # S-AFR: never sampled -> Root only
            al = torch.tensor([0.0, 0.0, 1.0], device=dev)
        else:
            stale = (a.rounds - 1) - last[cid]
            if stale > stale_tau:                            # S-AFR: stale -> fade towards Root
                fade = float(np.exp(-stale / stale_tau))
                al[0] *= fade; al[1] *= fade; al[2] = 1.0 - (al[0] + al[1])
        f = bb.extract_features(x)
        # Raw (unmasked) blended logits; the open/closed protocols are applied by Ctx.evaluate.
        return al[0] * local[cid](f) + al[1] * clus[cl_of[cid]](f) + al[2] * root(f)

    extra = {"min_participation": int(seen.min()), "mean_participation": round(float(seen.mean()), 2),
             "never_sampled": int((seen == 0).sum())}
    return c.evaluate(blend), extra


RUNNERS = {"FedAvg": run_fedavg, "FedRep": run_fedrep, "Ditto": run_ditto, "FedHEP (Ours)": run_fedhep}


# ----------------------------------------------------------------------------- driver
def run_job(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"seed{args.seed}_alpha{args.alpha}.json")
    res = {"config": {k: getattr(args, k) for k in (
        "dataset", "clients", "clients_per_round", "rounds", "local_epochs", "batch_size",
        "train_subset", "test_subset", "seed", "alpha")}, "methods": {}}
    if os.path.exists(out) and not args.force:
        res = json.load(open(out))
    device = torch.device(args.device) if args.device else detect_device()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    c = Ctx(args, device)
    _log(f"[scale50] seed={args.seed} alpha={args.alpha} device={device} rounds={args.rounds} E={args.local_epochs}")
    for name in METHODS:
        if name in res["methods"]:
            _log(f"  skip {name} (checkpointed)")
            continue
        t0 = time.time()
        (acc_open, acc_closed), extra = RUNNERS[name](c)
        res["methods"][name] = {"open": _summ(acc_open), "closed": _summ(acc_closed),
                                "seconds": round(time.time() - t0, 1), **extra}
        json.dump(res, open(out, "w"), indent=2)            # checkpoint after every method
        m = res["methods"][name]
        _log(f"  {name:14s} closed={m['closed']['mean']:.2f}/{m['closed']['bottom10']:.2f}  "
             f"open={m['open']['mean']:.2f}/{m['open']['bottom10']:.2f}  ({m['seconds']}s)")
    _log(f"[scale50] DONE -> {out}")


def merge():
    """Aggregate every outputs/scale50/seed*_alpha*.json into scale_50clients_results.json."""
    runs = {}
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "seed*_alpha*.json"))):
        d = json.load(open(f))
        runs.setdefault(str(d["config"]["alpha"]), []).append(d)
    merged = {}
    for alpha, items in runs.items():
        regime = REGIMES.get(alpha, f"alpha={alpha}")
        merged[regime] = {}
        for name in METHODS:
            rows = [d["methods"][name] for d in items if name in d["methods"]]
            if not rows:
                continue
            entry = {"n_seeds": len(rows), "seeds": [d["config"]["seed"] for d in items if name in d["methods"]]}
            for proto in ("closed", "open"):
                for stat in ("mean", "bottom10"):
                    v = np.array([r[proto][stat] for r in rows])
                    entry[f"{proto}_{stat}"] = round(float(v.mean()), 2)
                    entry[f"{proto}_{stat}_std"] = round(float(v.std()), 2)
            entry["mean"], entry["bottom10"] = entry["closed_mean"], entry["closed_bottom10"]  # legacy keys
            merged[regime][name] = entry
    out = os.path.join(_project_root, "outputs", "scale_50clients_results.json")
    json.dump(merged, open(out, "w"), indent=2)
    _log(f"merged {sum(len(v) for v in runs.values())} job files -> {out}")
    return merged


def run_50clients_scaling(num_clients=50, clients_per_round=10, num_rounds=40, batch_size=64, device=None,
                          train_subset=None, data_dir="./data", dataset="cifar100", skip_if_exists=False,
                          seeds=(42, 123, 7), job_timeout_s=1800):
    """Back-compat API for run_aamas_suite / run_runpod_suite.

    Each (seed, regime) runs in its OWN subprocess with a hard timeout, so a hang in one job can
    never stall (or silently burn GPU credit for) the rest of a long pipeline.
    """
    import subprocess
    out_path = os.path.join(_project_root, "outputs", "scale_50clients_results.json")
    if skip_if_exists and os.path.exists(out_path):
        return json.load(open(out_path))
    for seed in seeds:
        for alpha in REGIMES:
            cmd = [sys.executable, os.path.abspath(__file__), "--seed", str(seed), "--alpha", alpha,
                   "--rounds", str(num_rounds), "--clients", str(num_clients),
                   "--clients-per-round", str(clients_per_round), "--batch-size", str(batch_size),
                   "--dataset", dataset, "--data-dir", data_dir]
            if train_subset:
                cmd += ["--train-subset", str(train_subset)]
            if device:
                cmd += ["--device", str(device)]
            try:
                subprocess.run(cmd, check=True, timeout=job_timeout_s)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                _log(f"[scale50] job seed={seed} alpha={alpha} failed: {e!r} -- continuing")
    return merge()


def parse():
    p = argparse.ArgumentParser(description="50-client partial-participation benchmark")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=str, default="0.5", choices=sorted(REGIMES))
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--clients", type=int, default=50)
    p.add_argument("--clients-per-round", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100"])
    p.add_argument("--train-subset", type=int, default=15000)
    p.add_argument("--test-subset", type=int, default=3000)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true", help="ignore existing checkpoint")
    p.add_argument("--merge", action="store_true", help="merge finished jobs and exit")
    return p.parse_args()


if __name__ == "__main__":
    args = parse()
    merge() if args.merge else run_job(args)
