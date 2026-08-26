"""
Formal (epsilon, delta)-DP accounting for the HEP topology-routing channel.

Mechanism (client level, Channel 1 of the two-channel pipeline):
    1. Client computes head update Delta_i in R^{d_head}.
    2. L2 clip:          Delta_tilde = Delta_i * min(1, C_g / ||Delta_i||_2)
    3. JL sketch:        s_i = P @ Delta_tilde in R^m   (Achlioptas Rademacher, scaled 1/sqrt(m))
    4. Gaussian noise:   s_tilde_i = s_i + N(0, sigma^2 I_m)

Unit of privacy: client-level (adjacent datasets differ in one client's participation).
Under add/remove adjacency the clipped release has L2 sensitivity Delta_s <= C_g
(up to a multiplicative (1+beta_JL) JL norm-preservation slack; beta_JL is reported
empirically in Table XI.B).

Accountants implemented (no third-party DP dependencies):
    * Per-release closed-form Gaussian bound (Dwork & Roth 2014, Thm A.1):
            eps_rel = Delta_s * sqrt(2 * ln(1.25/delta)) / sigma
    * Basic composition over T rounds:        eps_basic = T * eps_rel
    * Renyi-DP composition of the Gaussian mechanism (Mironov 2017):
            eps_RDP(alpha) = T * alpha * Delta_s^2 / (2 * sigma^2)
            eps_rdp(delta) = min_{alpha>1} [ eps_RDP(alpha) + ln(1/delta)/(alpha-1) ]
    * Analytic Gaussian mechanism lower-bound note (Balle & Wang 2018) is cited
      in the paper as the tightest accountant; not reproduced here.

Outputs outputs/dp_budget.json consumed when regenerating Table XI.C.
"""

import argparse
import json
import math
import os

DEFAULT_SIGMAS = [0.01, 0.05, 0.10, 0.20]


def per_release_epsilon(sigma: float, c_g: float, delta: float) -> float:
    """Closed-form single-release Gaussian-mechanism bound."""
    return c_g * math.sqrt(2.0 * math.log(1.25 / delta)) / sigma


def basic_composition(eps_rel: float, rounds: int, delta: float) -> dict:
    return {"epsilon": rounds * eps_rel, "delta": min(1.0, rounds * delta)}


def rdp_composition(sigma: float, c_g: float, delta: float, rounds: int,
                    alphas=None) -> dict:
    """Optimal RDP conversion for T composed Gaussian releases."""
    if alphas is None:
        alphas = [1.0 + x / 100.0 for x in range(1, 10000)]
    best_eps, best_alpha = float("inf"), None
    for alpha in alphas:
        eps = rounds * alpha * (c_g ** 2) / (2.0 * sigma ** 2) \
            + math.log(1.0 / delta) / (alpha - 1.0)
        if eps < best_eps:
            best_eps, best_alpha = eps, alpha
    return {"epsilon": best_eps, "delta": delta, "alpha_star": round(best_alpha, 3)}


def sigma_for_target_epsilon(target_eps: float, c_g: float, delta: float) -> float:
    """Minimum single-round noise std achieving a target epsilon."""
    return c_g * math.sqrt(2.0 * math.log(1.25 / delta)) / target_eps


def main():
    ap = argparse.ArgumentParser(description="Auto-HEP routing-channel DP budget")
    ap.add_argument("--clip", type=float, default=1.0, help="L2 clipping threshold C_g")
    ap.add_argument("--delta", type=float, default=1e-5, help="per-release delta")
    ap.add_argument("--rounds", type=int, default=15, help="composition horizon T")
    ap.add_argument("--m", type=int, default=64, help="sketch dimension")
    ap.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    ap.add_argument("--out", type=str,
                    default=os.path.join("outputs", "dp_budget.json"))
    args = ap.parse_args()

    rows = []
    for sigma in args.sigmas:
        eps_rel = per_release_epsilon(sigma, args.clip, args.delta)
        comp = {
            "basic": basic_composition(eps_rel, args.rounds, args.delta),
            "rdp": rdp_composition(sigma, args.clip, args.delta, args.rounds),
        }
        rows.append({
            "sigma": sigma,
            "epsilon_per_release": round(eps_rel, 2),
            "epsilon_basic_T": round(comp["basic"]["epsilon"], 1),
            "epsilon_rdp_T": round(comp["rdp"]["epsilon"], 1),
            "rdp_alpha_star": comp["rdp"]["alpha_star"],
        })

    # Reference points quoted in the paper text.
    refs = {
        "sigma_for_eps10_single_round": round(
            sigma_for_target_epsilon(10.0, args.clip, args.delta), 3),
        "sigma_for_eps1_single_round": round(
            sigma_for_target_epsilon(1.0, args.clip, args.delta), 3),
        "sigma_for_eps5_rdp_T": round(
            math.sqrt(args.rounds / 2.0) * args.clip
            * math.sqrt(2.0 * math.log(1.0 / args.delta)) / 5.0, 3),
    }

    result = {
        "mechanism": "client-level Gaussian mechanism on m={m} JL sketch".format(m=args.m),
        "clip_C_g": args.clip,
        "delta_per_release": args.delta,
        "composition_rounds_T": args.rounds,
        "rows": rows,
        "reference_sigmas": refs,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    header = "{:>8} {:>18} {:>18} {:>18}".format(
        "sigma", "eps (release)", "eps (basic,T)", "eps (RDP,T)")
    print(header)
    print("-" * len(header))
    for r in rows:
        print("{:>8} {:>18} {:>18} {:>18}".format(
            r["sigma"], r["epsilon_per_release"],
            r["epsilon_basic_T"], r["epsilon_rdp_T"]))
    print("\nReference sigmas:", json.dumps(refs))
    print(f"\nSaved: {args.out}")
    return result


if __name__ == "__main__":
    main()
