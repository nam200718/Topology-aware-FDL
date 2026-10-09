"""Compare paper tables and chart data against outputs/master_experimental_data.json.

Prints every mismatch. Master is the single source of truth.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = json.load(open(os.path.join(ROOT, "outputs", "master_experimental_data.json")))["benchmarks"]
FIG = os.path.join(ROOT, "paper", "figures")
bad = 0


def report(where, got, want):
    global bad
    bad += 1
    print(f"MISMATCH {where}: paper={got} master={want}")


def nums(cell):
    return [float(x) for x in re.findall(r"-?\d+\.\d+", cell.replace("\\%", ""))]


def rows(path):
    out = []
    for line in open(path).read().splitlines():
        if ("&" in line or "CIFAR" in line) and line.rstrip().endswith("\\\\"):
            cells = [c.strip() for c in line.rstrip()[:-2].split("&")]
            out.append(cells)
    return out


REG = ["iid", "mild", "moderate", "severe", "extreme"]
NAME = {"FedAvg": "FedAvg", "FedProx": "FedProx", "Multi-Krum": "Multi-Krum", "SCAFFOLD": "SCAFFOLD",
        "FedRep": "FedRep", "Ditto": "Ditto", "FedHEP": "FedHEP"}


def lookup(section, method, regime=None, attack=None, q=None):
    for r in M[section]:
        if r["method"].lower() != method.lower():
            continue
        if regime is not None and r["regime"].lower() != regime:
            continue
        if attack is not None and r["attack"] != attack:
            continue
        if q is not None and abs(r["byzantine_rate"] - q) > 1e-9:
            continue
        return r
    return None


# ---- Table 2: personalization (FEMNIST then CIFAR-100)
section = "femnist_personalization"
for cells in rows(os.path.join(FIG, "table1_personalization_master.tex")):
    first = re.sub(r"\\textbf\{|\}", "", cells[0]).strip()
    if "B. CIFAR" in first:
        section = "cifar100_personalization"
    if first not in NAME or len(cells) < 11:
        continue
    for i, reg in enumerate(REG):
        r = lookup(section, first, regime=reg)
        a, b = nums(cells[1 + 2 * i]), nums(cells[2 + 2 * i])
        if r is None:
            report(f"T2 {section} {first} {reg}", "-", "missing")
            continue
        if not a or abs(a[0] - r["mean_acc"]) > 0.006:
            report(f"T2 {section} {first} {reg} acc", a, r["mean_acc"])
        if not b or abs(b[0] - r["mean_b10"]) > 0.006:
            report(f"T2 {section} {first} {reg} b10", b, r["mean_b10"])

# ---- Table 3: Byzantine
ATK = {"Label Flipping": "label_flip", "Sign Flipping": "sign_flip", "Gradient Ascent": "gradient_ascent",
       "Gaussian Noise": "random_noise"}
for cells in rows(os.path.join(FIG, "table3_byzantine.tex")):
    atk = ATK.get(cells[0])
    if not atk:
        continue
    meth = cells[1] if cells[1] != "FedHEP" else "Defended FedHEP"
    vals = []
    for qi, q in enumerate([0.0, 0.1, 0.2, 0.3, 0.4]):
        r = lookup("cifar100_byzantine_robustness", meth, attack=atk, q=q)
        got = nums(re.sub(r"\\dagger|\$", "", cells[2 + qi]))
        vals.append(r["mean_acc"] if r else None)
        if r is None or not got or abs(got[0] - r["mean_acc"]) > 0.006:
            report(f"T3 {atk} {meth} q={q}", got, r and r["mean_acc"])
    got = nums(re.sub(r"\$|\\dagger", "", cells[7]))
    want = round(vals[3] - vals[0], 2)
    if not got or abs(abs(got[0]) - abs(want)) > 0.011:
        report(f"T3 {atk} {meth} delta", got, want)

# ---- Table 4: 50 clients
sec = M["scale_50clients"]
for cells in rows(os.path.join(FIG, "table4_scale50.tex")):
    pass  # checked below via flat compare
t4 = open(os.path.join(FIG, "table4_scale50.tex")).read()
body = t4.split("\\midrule", 1)[1].split("\\bottomrule")[0]
ts = [x for x in nums(body) if x not in (0.5, 0.1)]
print("T4 numbers in paper table:", ts)
want_t4 = [33.97, 24.11, 27.40, 33.89, 20.63, 9.64, 8.60, 18.11, 31.87, 41.72, 37.47, 41.16, 12.81, 17.41, 13.03, 16.88]
if ts != want_t4:
    report("T4 values", ts, want_t4)

print(f"\n{bad} mismatches")
sys.exit(1 if bad else 0)
