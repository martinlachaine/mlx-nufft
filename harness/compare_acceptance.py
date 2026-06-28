"""Compare two acceptance runs (e.g. M1 mini vs M-series Max).

Usage: compare_acceptance.py results/acceptance.json other/acceptance.json
Prints per-config GPU time ratios (a bandwidth-scaling record across machines)
and accuracy deltas (should be ~equal: accuracy is machine-independent fp32
arithmetic).
"""

import sys
import json

if __name__ == "__main__":
    a = json.load(open(sys.argv[1]))
    b = json.load(open(sys.argv[2]))
    print(f"A: {a['machine']}")
    print(f"B: {b['machine']}")
    bi = {(r["geom"], r["N"], r["P"]): r for r in b["rows"]}
    print("\n| geometry | N | P | A gpu s | B gpu s | B/A speedup | "
          "A rel-L2 | B rel-L2 |")
    print("|---|---|---|---|---|---|---|---|")
    ratios = []
    for r in a["rows"]:
        k = (r["geom"], r["N"], r["P"])
        if k not in bi:
            continue
        o = bi[k]
        ratio = r["t_gpu"] / o["t_gpu"]
        ratios.append(ratio)
        print(f"| {k[0]} | {k[1]} | {k[2]:.0e} | {r['t_gpu']:.2f} | "
              f"{o['t_gpu']:.2f} | {ratio:.2f}x | "
              f"{r['rel_l2']:.2e} | {o['rel_l2']:.2e} |")
    if ratios:
        import statistics
        print(f"\nmedian speedup B over A: {statistics.median(ratios):.2f}x "
              f"(min {min(ratios):.2f}x, max {max(ratios):.2f}x)")
