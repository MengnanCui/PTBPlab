"""Parameter sets: where the Slater-Koster data for each type label comes from.

A :class:`ParamSet` describes a directory of SKF files (plain ``A-B.skf`` or
gzip-compressed ``A-B.skf.gz``) and the options that turn them into an
:class:`~pydftb.skf.SKSet` for a given list of type labels.  Labels are file
name stems: ``"O"`` reads ``O-H.skf``; a second oxygen type ``"O2"`` would
read ``O2-H.skf`` (DFTB+/ASE ``Type2FileNames`` convention), which is how one
element gets several parameter sets.
"""
from __future__ import annotations

import glob
import os

from .skf import SKSet


class ParamSet:
    """
    Parameters
    ----------
    skdir : directory containing the SKF files.
    max_l : optional {label: l_max}; missing labels use the ASE/PTBP rule
            (highest shell with non-zero occupation in the homonuclear SKF).
    template : file-name template; default detects ``{a}-{b}.skf`` or ``.skf.gz``.
    shell_resolved : orbital-resolved SCC (DFTB+ ``ShellResolvedSCC``).
    hubbard_from : non-shell-resolved U source shell ("s" = DFTB+ behaviour).
    """

    def __init__(self, skdir: str, max_l: dict[str, int] | None = None,
                 template: str | None = None, shell_resolved: bool = False,
                 hubbard_from: str = "s"):
        if not os.path.isdir(skdir):
            raise FileNotFoundError(f"SKF directory not found: {skdir}")
        self.skdir = skdir
        self.max_l = dict(max_l or {})
        self.template = template or self._detect_template(skdir)
        self.shell_resolved = shell_resolved
        self.hubbard_from = hubbard_from
        self._cache: dict[tuple, SKSet] = {}

    @staticmethod
    def _detect_template(skdir: str) -> str:
        if glob.glob(os.path.join(skdir, "*-*.skf")):
            return "{a}-{b}.skf"
        if glob.glob(os.path.join(skdir, "*-*.skf.gz")):
            return "{a}-{b}.skf.gz"
        raise FileNotFoundError(f"no '*-*.skf' or '*-*.skf.gz' files in {skdir}")

    def labels_available(self) -> list[str]:
        """Labels that have a homonuclear file ``X-X.skf`` in the directory."""
        suffix = self.template.format(a="", b="")[1:]          # ".skf" / ".skf.gz"
        out = []
        for path in glob.glob(os.path.join(self.skdir, "*" + suffix)):
            stem = os.path.basename(path)[: -len(suffix)]
            a, _, b = stem.partition("-")
            if a == b:
                out.append(a)
        return sorted(out)

    def skset(self, labels) -> SKSet:
        """SKSet for exactly these labels (cached; SKF parsing is cached too)."""
        key = tuple(sorted(set(labels)))
        if key not in self._cache:
            for a in key:
                for b in key:
                    path = os.path.join(self.skdir, self.template.format(a=a, b=b))
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"missing Slater-Koster file {path}")
            self._cache[key] = SKSet(self.skdir, list(key),
                                     {a: self.max_l[a] for a in key if a in self.max_l},
                                     template=self.template, shell_resolved=self.shell_resolved,
                                     hubbard_from=self.hubbard_from)
        return self._cache[key]

    def __repr__(self) -> str:
        return (f"ParamSet({self.skdir!r}, template={self.template!r}, "
                f"shell_resolved={self.shell_resolved})")
