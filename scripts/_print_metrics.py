import json, numpy as np
with open("hrl_vs_flat_vs_gradient_50ep.json") as f:
    data = json.load(f)
hrl, grad = data["beefi_hrl"], data["gradient_policy"]
def a(eps, key): return np.mean([e[key] for e in eps])
def r(eps, n, d): return np.mean([e[n]/e[d] if e[d]>0 else 0 for e in eps])
print("Axis               HRL    Grad   Delta")
print("-"*44)
for name, h, g in [
    ("Harvest Rate", a(hrl,"harvest_rate"), a(grad,"harvest_rate")),
    ("Hard Window", r(hrl,"hard_done","hard_total"), r(grad,"hard_done","hard_total")),
    ("Soft Window", r(hrl,"soft_done","soft_total"), r(grad,"soft_done","soft_total")),
    ("No-Window", r(hrl,"none_done","none_total"), r(grad,"none_done","none_total")),
    ("Sat Survival", a(hrl,"alive_sats")/25, a(grad,"alive_sats")/25),
]:
    print(f"{name:<18s} {h:.3f}  {g:.3f}  +{h-g:.3f}")
