"""A1 data for Li and Li-O: tagged copies of the M5c/M5d VASP data (examples/07/data).

    python prepare_data.py [--out data] [--d2 D2_COLLECTED.xyz ...]

Every frame gets ``info``:

* ``name``: unique;
* ``family``: the group of examples/07, e.g. eos_li_bcc, rat_li16o, dimer_LiO, iface;
* ``role``: train, test or ood;
* ``kspacing``: 0.15 for bulk, 0.25 for the interface (as computed), ``gamma`` for the dimers
  in their box;
* ``source`` and ``dft``: the origin and the settings.

The roles (A0 section 3, A1.1 and A1.2):

| set  | train | test | ood |
|---|---|---|---|
| Li   | eos_li_bcc, eos_li_fcc, rat_li_bcc16 (seeds 1-3), dimer_Li2 | rat_li_bcc16 seed 101 | cmp_li_bcc (0.70-0.85 V0) |
| Li-O | train.xyz (68) + short_range.xyz (43) | test.xyz (8) | interface.xyz (3) |

Data from batch D2 (``make_d2.py``, once computed and collected) is added with ``--d2``; its
frames keep the roles of D2's manifest (``info['role']``).  Writes ``li_{train,test,ood}.xyz``,
``li_o_{train,test,ood}.xyz``, ``E0s_dft.json`` and ``manifest.csv`` in ``--out``.
"""
import argparse
import csv
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.paths import EX07_DATA, setup  # noqa: E402

setup()
import ase.io  # noqa: E402

DFT = "VASP 6.6.0 PBE (potpaw_PBE.54 Li_sv, O), ENCUT 800, PREC Accurate, ISMEAR 0, SIGMA 0.05, ISPIN 1"


def tag(frames, role, source, kspacing=None):
    out = []
    for a in frames:
        a.info["name"] = a.info.get("name") or a.info["structure_name"]
        a.info["family"] = a.info.get("family") or a.info["group"]
        a.info["role"] = role
        fam = a.info["family"]
        if kspacing is not None:
            a.info["kspacing"] = kspacing
        elif fam.startswith("dimer_"):
            a.info["kspacing"] = "gamma"
        else:
            a.info.setdefault("kspacing", 0.15)
        a.info["source"] = source
        a.info["dft"] = DFT
        out.append(a)
    return out


def interface(frames):
    """The interface runs used KSPACING 0.25 = a 4 x 4 x 1 grid (make_iface_dft.py, M5c); the
    rule gives 2 along c for this cell (2 pi |b3| / 0.25 = 1.01), so the grid is set explicitly."""
    for a in frames:
        a.info["kpts"] = "4x4x1"
    return frames


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--src", default=EX07_DATA, help="examples/07_battery_li_li2o/data")
    ap.add_argument("--d2", nargs="*", default=[], help="collected D2 frames (with info role)")
    args = ap.parse_args(argv)
    from utils.runlog import session
    with session("prepare_data", vars(args), root=os.path.join(args.out, "logs")) as log:
        src = args.src

        def load():
            """Fresh, tagged frames (the Li and Li-O sets must not share objects)."""
            read = lambda f: ase.io.read(os.path.join(src, f), ":")       # noqa: E731
            d2 = []
            for p in args.d2:
                for a in ase.io.read(p, ":"):
                    d2 += tag([a], a.info.get("role", "ood"), os.path.basename(p))
            return (tag(read("train.xyz"), "train", "examples/07/data/train.xyz"),
                    tag(read("test.xyz"), "test", "examples/07/data/test.xyz"),
                    tag(read("short_range.xyz"), "train", "examples/07/data/short_range.xyz"),
                    interface(tag(read("interface.xyz"), "ood", "examples/07/data/interface.xyz",
                                  kspacing=0.25)), d2)

        train, test, short, iface, d2 = load()
        sets = {
            "li_o_train": train + short + [a for a in d2 if a.info["role"] == "train"],
            "li_o_test": test + [a for a in d2 if a.info["role"] == "test"],
            "li_o_ood": iface + [a for a in d2 if a.info["role"] == "ood"],
        }
        train, test, short, iface, d2 = load()
        li = lambda frames: [a for a in frames if set(a.get_chemical_symbols()) == {"Li"}]  # noqa: E731
        sets["li_train"] = ([a for a in li(train + short) if not a.info["family"].startswith("cmp_")]
                            + [a for a in li(d2) if a.info["role"] == "train"])
        sets["li_test"] = li(test) + [a for a in li(d2) if a.info["role"] == "test"]
        sets["li_ood"] = ([a for a in li(short) if a.info["family"].startswith("cmp_")]
                          + [a for a in li(d2) if a.info["role"] == "ood"])
        for name in ("li_train", "li_test", "li_ood"):
            for a in sets[name]:
                a.info["role"] = name.split("_", 1)[1]
        os.makedirs(args.out, exist_ok=True)
        rows = []
        for name, frames in sets.items():
            names = [a.info["name"] for a in frames]
            assert len(names) == len(set(names)), f"repeated names in {name}"
            path = os.path.join(args.out, f"{name}.xyz")
            ase.io.write(path, frames, format="extxyz")
            log.event("wrote", set=name, n=len(frames), path=path)
            for a in frames:
                rows.append({"set": name, "name": a.info["name"], "family": a.info["family"],
                             "role": a.info["role"], "natoms": len(a),
                             "formula": a.get_chemical_formula(), "kspacing": a.info["kspacing"],
                             "source": a.info["source"]})
        with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        shutil.copy(os.path.join(src, "E0s_dft.json"), os.path.join(args.out, "E0s_dft.json"))
        counts = {k: len(v) for k, v in sets.items()}
        log.save_json("counts.json", counts)
        print(json.dumps(counts, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
