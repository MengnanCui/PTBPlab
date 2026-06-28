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
        '--optimization_option', help="Options: EnergyGeometry, BandStructure, Dataset, Reaction",
        type=str,
        choices=["EnergyGeometry", "energygeometry",
                 "BandStructure",  "bandstructure",
                 "Dataset"      ,  "dataset",
                 "Reaction"     ,  "reaction",
        ],
        default=None,
    )

    parser.add_argument(
        '--parameters', help='Choose the parameters you want to optimize',
        nargs='+', type=str, default=None,
    )

    parser.add_argument(
        '--multi_element', help='Give the element symbols that you want to optimize among the composition',
        nargs='+',
        type=str,
        default=None,
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
                          "Default: 11 for energygeometry/dataset/reaction, "
                          "100 for bandstructure.",
        type=int,
        default=None,
    )
    parser.add_argument(
        '--optimizer', help="Outer-loop optimiser: 'bo' (sequential Gaussian-process "
                            "BO via skopt; default), 'parallel_bo' (skopt.Optimizer "
                            "with batched proposals evaluated in parallel via mp.Pool), "
                            "or 'pso' (particle swarm via pyswarms).",
        type=str,
        choices=["bo", "parallel_bo", "pso"],
        default="bo",
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

    # args_input = parser.parse_args()
    return parser.parse_args()


        
