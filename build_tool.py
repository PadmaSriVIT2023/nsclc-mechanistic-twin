"""build_tool.py - Embed the frozen model export into the research tool page."""
import json, pathlib, re, sys
T = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(T.parent / "manuscript"))
from ms_numbers import build
N = build(); m = json.dumps(json.load(open(T / "model_export.json")), separators=(",", ":"))
s = (T / "tool.tmpl.html").read_text()
rep = {"__MODEL__": m, "__N__": str(N["tcga_n"]), "__EV__": str(N["tcga_ev"]),
       "__C_ONC__": f"{N['onc_clintwin_coxph']} (95% CI {N['onc_clintwin_coxph_ci'].replace('--', '–')})",
       "__C_NCI__": f"{N['nci_clintwin_coxph']} (95% CI {N['nci_clintwin_coxph_ci'].replace('--', '–')})"}
for k, v in rep.items(): s = s.replace(k, v)
assert not re.search(r"__(MODEL|N|EV|C_ONC|C_NCI)__", s)
(T / "nsclc_twin_tool.html").write_text(s); print("tool ok", len(s))
