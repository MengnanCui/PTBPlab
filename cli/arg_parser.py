# Pasing functionalities
# Authors: Mengnan Cui,imitate from MACE

import argparse
import ast
from typing import List, Optional, Union

def build_default_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()


    # Name and seed
    parser.add_argument("--name", help="experiment name", default='PTBP-OPTIMIZATION')
    parser.add_argument("--seed", help="random seed", type=int, default=123)

    # Directories
    parser.add_argument(
        "--ref_dir", help="The folder includes reference files",
        type=str,
        default="dft"
    )
    parser.add_argument(
        "--results_dir", help="The folder includes results of optimization",
        type=str,
        default="results",
    )

    # Device and logging
    parser.add_argument(
        "--log_file", help="The log file name",
        type=str,
        default="log.out",
    )

    parser.add_argument(
        "--para_file", help="The parameter file name",
        type=str,
        default="par.out",
    )



    parser.add_argument(
        '--optimization_option', help="Options: EnergyGeometry, BandStructure, Dataset, Reaction, "
                                      "Bands (the A-series band loss alone, pydftb; needs "
                                      "--band_refs)",
        type=str,
        choices=["EnergyGeometry", "energygeometry",
                 "BandStructure",  "bandstructure",
                 "Dataset"      ,  "dataset",
                 "Reaction"     ,  "reaction",
                 "Bands"        ,  "bands",
        ],
        default=None,
    )

    parser.add_argument(
        '--parameters', help='Choose the parameters you want to optimize',
        nargs='+', type=str, default=None,
    )

    parser.add_argument(
        '--sigma_range', help='Bounds of the inner sigma_rep search (M5c; default 0.1 0.8)',
        nargs=2, type=float, default=None,
    )
    parser.add_argument(
        '--multi_element', help='Fit only these elements (symbols); the other elements keep '
                                'the (r0_w, r0_d, sigma_rep) PTBP published (M5b)',
        nargs='+',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--shared_parameters', help="One (r0_w, r0_d, sigma_rep) for all elements, as before "
                                    "M5b; default: one set per element (type), the penalty "
                                    "being the geometric mean of the sets' penalties.  With "
                                    "--multi_element the listed elements share one set and the "
                                    "others keep PTBP's; a --multi_element of every element of "
                                    "the dataset is refused",
        action='store_true',
        default=None,
    )
    parser.add_argument(
        '--types', help="Several types of one element that switch with the coordination "
                        "number (M5b), as JSON: a list of {\"element\": \"Cu\", \"labels\": "
                        "[\"Cu2\", \"Cu\"], \"centre\": 11.0, \"width\": 1.0[, \"cutoff\": ..., "
                        "\"start\": {label: [r0_w, r0_d]}]}, the labels in order of increasing "
                        "coordination number; needs --engine pydftb.  With the energygeometry "
                        "loss a type may get no data (a warning is printed): prefer reaction "
                        "or dataset",
        type=str,
        default=None,
    )
    parser.add_argument(
        '--type_start', help="Starting point (r0_w r0_d, Bohr) of a type's parameters: "
                             "LABEL R0_W R0_D (repeatable)",
        nargs=3, action='append', metavar=('LABEL', 'R0_W', 'R0_D'),
        default=None,
    )
    parser.add_argument(
        '--confinement', help="hotcent's confinement scheme for the confined atoms: "
                              "'selfconsistent' (default; hotcent 1.x, r0_w shapes the "
                              "orbitals and r0_d the density) or 'perturbative' (hotcent "
                              ">= 2.0 default, with which r0_d has no effect; reproduces "
                              "runs before M4e)",
        choices=['selfconsistent', 'perturbative'],
        default='selfconsistent',
    )
    parser.add_argument(
        '--superposition', help="Type of the superposition",
        type=str,
        default=None,
        choices=[
            "density",
            "potential",
        ]
    )


    parser.add_argument(
        '--E0s',
        help="The atomic energies for reference calculator(DFT), it's usefull only when the optimization_option is Dataset",
        type=str,
        default=None,
    )

    # skf generator
    parser.add_argument(
        "--skf_generator", help="Whether generate the skf files",
        type=str,
        default=None,
        choices=[
            "FULL", "full",
            "BAND", "band",
            "REP", "rep",
        ]
    )

    parser.add_argument(
        "--symbols",
        help="The symbols of the elements",
        nargs='+',
        type=str,
        default=None,
    )

    parser.add_argument(
        '--confinement_parameters', 
        help="The confinement parameter for the superposition, r0_w, r0_d, p",
        nargs='+',
        type=float,
        default=None,
    )

    parser.add_argument(
        '--repulsive_parameters',
        help="The repulsive parameters for the superposition, sigma_rep, kxc",
        nargs='+',
        type=float,
        default=None,
    )

    parser.add_argument(
        '--known_parameters',
        help="The known parameters for the superposition, PTBP, QNplusREP, QUASINANO2013, PRIOR",
        type=str,
        default=None,
        choices=[
            "PTBP", "ptbp",
            "QNplusREP", "qnplusrep",
            "QUASINANO2013", "quasinano2013",
            "PRIOR", "prior",
            "PERSONAL", "personal",
        ]
    )



    # Defaults
    parser.add_argument(
        '--checkpoint', help="The checkpoint file name",
        type=str,
        default="checkpoint.pkl",
    )
    parser.add_argument(
        '--n_calls', help="Total number of BO iterations (gp_minimize n_calls). "
                          "Default for energygeometry/dataset/reaction: 11 for up to 2 "
                          "optimised dimensions (one parameter set), 10 + 5 x dims above "
                          "(two elements: 30); 100 for bandstructure.  An explicit value "
                          "always wins.",
        type=int,
        default=None,
    )
    parser.add_argument(
        '--optimizer', help="Outer-loop optimiser: 'bo' (sequential Gaussian-process "
                            "BO via skopt; default), 'parallel_bo' (skopt.Optimizer "
                            "with batched proposals evaluated in parallel via mp.Pool), "
                            "'pso' (particle swarm via pyswarms), 'gradient' (projected "
                            "BFGS on the loss gradient; needs --engine pydftb and torch) "
                            "or 'bo_grad' (BO, then a few gradient steps).",
        type=str,
        choices=["bo", "parallel_bo", "pso", "gradient", "bo_grad"],
        default="bo",
    )
    parser.add_argument(
        '--grad_maxiter', help="Iterations of the gradient optimiser. Default: 20 for "
                               "'gradient', 5 for the refinement of 'bo_grad'.",
        type=int,
        default=None,
    )
    parser.add_argument(
        '--hotcent_fd_step', help="Steps (Bohr) of the hotcent central differences of the "
                                  "gradient: r0_w [r0_d]. Default: 0.04 0.2.",
        type=float,
        nargs='+',
        default=None,
    )
    parser.add_argument(
        '--n_particles', help="Population size for PSO and the batch width for "
                              "parallel_bo. Default: 4. Total evaluations are "
                              "n_particles * (n_calls / n_particles) for PSO "
                              "and n_calls for parallel_bo.",
        type=int,
        default=4,
    )
    parser.add_argument(
        '--ncores', help="Cores for the parallel BO step: a number, or 'auto' (default): "
                         "every core this process may use (CPU affinity, cgroup quota, "
                         "SLURM_CPUS_PER_TASK, PTBP_NCORES). 1 runs serially.",
        type=str,
        default='auto',
    )
    parser.add_argument(
        '--postprocess_only', help="Skip the optimisation and reuse a previous "
                                   "run's result.pkl in cwd: re-emit "
                                   "convergence.png, copy the best opt_<r0_w>_"
                                   "<r0_d>/ folder, repack results.tgz. Useful "
                                   "when a long optimisation succeeded but "
                                   "post-processing failed.",
        action='store_true',
    )
    parser.add_argument(
        '--xc', help="The Exchange Correlation method",
        type=str,
        default='GGA_X_PBE+GGA_C_PBE',
    )
    parser.add_argument(
        '--kpt_density', help="The kpoint density for DFTB calculation",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        '--engine', help="DFTB engine: dftbplus (DFTB+ through ASE, default) or "
                         "pydftb (in-process Python engine, see pydftb/README.md)",
        choices=['dftbplus', 'pydftb'],
        default='dftbplus',
    )
    parser.add_argument(
        '--smearing', help="Electronic smearing for both engines. 'none' (default) "
                           "reproduces PTBP's historical setting: DFTB+ ignored the "
                           "2000 K Methfessel-Paxton keywords, i.e. Fermi at T = 0.",
        choices=['none', 'fermi', 'gaussian', 'mp'],
        default='none',
    )
    parser.add_argument(
        '--smearing_temperature', help="Electronic temperature in K (if --smearing != none)",
        type=float,
        default=2000.0,
    )
    parser.add_argument(
        '--mp_order', help="Methfessel-Paxton order (if --smearing mp)",
        type=int,
        default=2,
    )
    parser.add_argument(
        '--pydftb_forces', help="pydftb force derivatives: 'dftbplus' (default) reproduces "
                                "DFTB+'s 0.01 Bohr finite differences, 'exact' uses exact derivatives",
        choices=['dftbplus', 'exact'],
        default='dftbplus',
    )
    parser.add_argument(
        '--max_l', help="MaxAngularMomentum of every element, both engines: 'occupied' "
                        "(default) = the highest shell occupied in the neutral atom, PTBP's "
                        "basis; 'highest' = every shell the SKF has integrals for",
        choices=['occupied', 'highest'],
        default='occupied',
    )
    parser.add_argument(
        '--pydftb_backend', help="pydftb arrays: 'numpy' (default) or 'torch' (PyTorch, on "
                                 "the CPU or a GPU, see --device)",
        choices=['numpy', 'torch'],
        default='numpy',
    )
    parser.add_argument(
        '--device', help="torch device of --pydftb_backend torch: cpu (default), auto, cuda, "
                         "cuda:1, mps, xpu. On a GPU the DFTB stage runs in the main process.",
        type=str,
        default='cpu',
    )
    parser.add_argument(
        '--dtype', help="precision of --pydftb_backend torch: float64 (default) or float32",
        choices=['float64', 'float32'],
        default='float64',
    )
    parser.add_argument(
        '--dft_file', help="The XYZ file that stores reference structure and info",
        type=str,
        default='dft.xyz',
    )

    parser.add_argument(
        '--dft_band_file', help="The XYZ file that stores reference structure and info",
        type=str,
        default='dft.json',
    )

    parser.add_argument(
        '--eos_file', help="The file that sotres EOS fitting for dft",
        type=str,
        default='fit.json',
    )
    parser.add_argument(
        '--eos_points', help="The number of points for EOS fitting",
        type=int,
        default=11,
    )

    # A series (pydftb/docs/design/A0-accuracy-programme.html, section 8); every default is
    # the loss and the calculation of before, bit for bit
    parser.add_argument(
        '--energy_reduction', help="dataset/reaction: mu_E = sum_i w_i |dE_i| with w_i = 1 "
                                   "('sum', default, as before), 1/n ('mean') or 1/(G n_g) "
                                   "('group': every group counts the same; the group of a "
                                   "structure is its info 'family', 'config_type' or 'group', "
                                   "else its formula)",
        choices=['sum', 'mean', 'group'],
        default='sum',
    )
    parser.add_argument(
        '--force_weight', help="dataset/reaction: c in mu = ((1 - c) mu_E + c mu_F) P "
                               "(default 0.1, as before)",
        type=float,
        default=None,
    )
    parser.add_argument(
        '--energy_offset', help="dataset/reaction: 'per-element' removes per-element energy "
                                "offsets (eV/atom, weighted least squares) from the "
                                "formation-energy errors before mu_E; 'none' (default) keeps "
                                "the absolute errors",
        choices=['none', 'per-element'],
        default='none',
    )
    parser.add_argument(
        '--band_weight', help="Adds W L_band inside the penalty (every mode): the band loss of "
                              "pydftb.bandloss on --band_refs at the step's SK files, pydftb "
                              "engine (default 0: no band term)",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        '--band_refs', help="BandRef JSON files or glob patterns (pydftb.bandref) for "
                            "--band_weight and --optimization_option bands",
        nargs='+', type=str, default=None,
    )
    parser.add_argument(
        '--band_settings', help="pydftb.bandloss.BandSettings as JSON, e.g. '{\"levels\": [1, 2], "
                                "\"conduction_weight\": 0.4}' (default: the A-series settings)",
        type=str,
        default=None,
    )
    parser.add_argument(
        '--fixed_r0', help="Freeze the electronic part at these radii (Bohr: r0_w r0_d of every "
                           "parameter set) and fit sigma_rep only (the S3 route of the A series)",
        nargs='+', type=float, default=None,
    )
    parser.add_argument(
        '--sigma_finish', help="After the optimiser, a finer sigma_rep search at the best point "
                               "on its step files (sigma_finish.json; the best folder's SKF "
                               "files get the new sigma if the loss is lower)",
        action='store_true',
        default=None,
    )
    parser.add_argument(
        '--kspacing', help="Every periodic structure gets VASP's Gamma-centred KSPACING grid "
                           "N_i = max(1, ceil(2 pi |b_i| / KSPACING)) (1/A) instead of the "
                           "--kpt_density rule (default: off)",
        type=float,
        default=None,
    )
    parser.add_argument(
        '--energy_kind', help="'total' (default): DFTB+'s Total energy; 'sigma0': Total - TS/2, "
                              "VASP's energy(sigma->0) (pydftb engine)",
        choices=['total', 'sigma0'],
        default='total',
    )
    parser.add_argument(
        '--third_order', help="DFTB3 (pydftb engine): 'full' (ThirdOrderFull) or 'onsite'; "
                              "needs --hubbard_derivs",
        choices=['full', 'onsite'],
        default=None,
    )
    parser.add_argument(
        '--hubbard_derivs', help="Hubbard derivatives for --third_order as JSON {element: Ud "
                                 "(Hartree/e)}",
        type=str,
        default=None,
    )
    parser.add_argument(
        '--hubbard', help="Hubbard values replacing the SKF ones as JSON {element: U (Hartree)} "
                          "(pydftb engine)",
        type=str,
        default=None,
    )

    # args_input = parser.parse_args()
    return parser.parse_args()


        
