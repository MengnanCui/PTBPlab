"""Textual-based setup wizard for `ptbp` (the `ptbp setup` subcommand).

The wizard is *setup-only*: it walks the user through dataset selection,
shallow validation, mode/optimizer/E0s choice, then writes a
`ptbp_run.yaml` and prints the equivalent `ptbp ...` command. Long-running
optimisation is left to the regular CLI; the user runs the printed
command themselves.

Designed for the constrained environment we expect in practice:
plain SSH + a normal terminal (no X-forwarding, no port-forwarding).
Textual writes ANSI to stdout/stdin so this works anywhere ssh works.
"""
