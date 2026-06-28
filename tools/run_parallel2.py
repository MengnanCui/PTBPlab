import os, math, sys
import ase.io
import argparse
from argparse import RawTextHelpFormatter


def seperate_atoms(dataset_infile, num_parts, tag):

    dataset = ase.io.read(dataset_infile, ':')
    dataset_file = dataset_infile.strip().split('/')[-1]
    dataset_file_output = dataset_file.replace('.xyz', f'_{tag}.xyz')
    num_atoms = len(dataset)
    

    # seperate val_dataset into num_parall parts
    size_subs = math.ceil(num_atoms / num_parts)

    subdataset_names = [f"{tag}_{i}/tmp.{dataset_file.split('.')[0]}.xyz" for i in range(num_parts)]
    # subdataset_gap_names = [f"{tag}_{i}/tmp.{dataset_file.split('.')[0]}_dftb.xyz" for i in range(num_parts)]
    # print(f"subdataset_names: {'  '.join(subdataset_names)}")
    # print(f"subdataset_gap_names: {'  '.join(subdataset_gap_names)}")


    # write subdataset
    # cmd_file = open("cmd.lst", 'w')
    for n in range(num_parts):
        start_index = n * size_subs
        end_index = (n + 1) * size_subs
        subdataset = dataset[start_index: end_index]

        subfolder_name = f"{tag}_{n}"
        if not os.path.exists(subfolder_name):
            os.system(f"mkdir {subfolder_name}")
            # os.system(f"cp {calculator_code} {subfolder_name}")
            # if tag.lower() != "xtb":
            #     os.system(f"cp input/*.skf {subfolder_name}")
        # else:
            # os.system(f"cp {calculator_code} {subfolder_name}")
            # if tag.lower() == "dftb":
                # os.system(f"cp *.skf {subfolder_name}")
        subdataset_name = subdataset_names[n]  
        ase.io.write(subdataset_name, subdataset)
    return subdataset_names
        
def prep_gnuparallel(dataset_infile:str, num_parall, calculator_code:str, tag:list, submit_dict=None):
    """
    Be parallelized !!!

    Parameters:
    ----------
    dataset_file: str
        path to the validation dataset file
    num_parall: int
        number of parallel processes
    tag: list
        [static/xtb1/xtb2, kpts_dens] tag for DFTB calculation
    """

    # 1. Divide input file into num_parall folders
    # initialize information
    dataset = ase.io.read(dataset_infile, ':')
    dataset_file = dataset_infile.strip().split('/')[-1]
    dataset_file_output = dataset_file.replace('.xyz', f'_dftb.xyz')
    num_atoms = len(dataset)
    

    # seperate val_dataset into num_parall parts
    size_subs = math.ceil(num_atoms / num_parall)

    subdataset_names = [f"dftb_{i}/tmp.{dataset_file.split('.')[0]}.xyz" for i in range(num_parall)]
    subdataset_gap_names = [f"dftb_{i}/tmp.{dataset_file.split('.')[0]}_dftb.xyz" for i in range(num_parall)]
    print(f"subdataset_names: {'  '.join(subdataset_names)}")
    print(f"subdataset_gap_names: {'  '.join(subdataset_gap_names)}")


    # write subdataset
    cmd_file = open("cmd.lst", 'w')
    for n in range(num_parall):
        start_index = n * size_subs
        end_index = (n + 1) * size_subs
        subdataset = dataset[start_index: end_index]

        subfolder_name = f"dftb_{n}"
        if not os.path.exists(subfolder_name):
            os.system(f"mkdir {subfolder_name}")
            os.system(f"cp ../{calculator_code} {subfolder_name}")
            if tag[0].lower()[:3] != "xtb":
                os.system(f"cp ../input/*.skf {subfolder_name}")
        else:
            os.system(f"cp ../{calculator_code} {subfolder_name}")
        subdataset_name = subdataset_names[n]  
        ase.io.write(subdataset_name, subdataset)


        # write cmd file to cmd.lst, here i run calculator with python
        cmd_run = f"python  {calculator_code} -sp tmp.{dataset_file.split('.')[0]}.xyz {tag[0]} -d {tag[-1]}"

        cmd_file.write(f"cd {subfolder_name} ; {cmd_run} >> output\n")
    cmd_file.close()

    print("Seperate validation dataset into " + str(num_parall) + " parts for parallel calculation.")


    # # 2. Run GAP on each part
    # os.system(f"parallel --delay 0.2 --joblog task.log --progress --resume -j {num_parall} < cmd.lst")


    # # 3. Collect data after finished
    # os.system(f"cat {' '.join(subdataset_gap_names)} > {dataset_file_output}")
    # os.system("mv task.log task_backup.log")

    gnuparall_system(subdataset_gap_names, dataset_file_output, num_parall, submit_dict)

    os.system(f'sbatch submit.sh')


    # output_dataset = ase.io.read(dataset_file_output, ':')
    # if len(output_dataset) == len(dataset):
    #     os.system(f"rm -r {tag}_*")
    # else:
    #     print(f"WARNING: len(ref_dataset):{len(dataset)} != len(gap_dataset):{len(output_dataset)}")



def gnuparall_system(sub_names, output_names, num_parall, submit_dict):

# 2. Run GAP on each part
    parall = f"parallel --delay 0.2 --joblog task.log --progress --resume -j {num_parall} < cmd.lst; "
    catt = f"cat {' '.join(sub_names)} > {output_names} && rm -r dftb_*"
    mvv = "mv task.log task_backup.log"

    if submit_dict == None:
        os.system(parall)



    else:   
        run_command = f"{parall} \n{catt} \n{mvv}"
        mpcdf_submit(time=submit_dict['time'],
                    disk_space=submit_dict['disk_space'],
                    job_name=submit_dict['job_name'],
                    partition=submit_dict['partition'],
                    core=submit_dict['core'],
                    run_command=run_command)

        # raven_submit(time="23:00:00", 
    #              disk_space=90000, 
    #              job_name=f"DFTB",
    #              partition="general", 
    #              core = core,
    #              run_command=run_command)

    # os.system(f"parallel --delay 0.2 --joblog task.log --progress --resume -j {num_parall} < cmd.lst")


    # # 3. Collect data after finished
    # os.system(f"cat {' '.join(sub_names)} > {output_names}")
    # os.system("mv task.log task_backup.log")
    

def mpcdf_submit(time:str="23:59:00", 
                 disk_space:int=100000,
                 job_name:str="gap-fit",
                 partition:str="middle",
                 core:int=40,
                 run_command:str="python run.py -gap -val -err > py.out"):
    """
    Write submition script for cobra

    Parameters:
    ----------
    time(str): time for job
        e.g. "23:59:00"

    disk_space(int): disk space for job
        e.g. 100000
    
    job_name(str): job name
        e.g. "gap-fit"

    Returns:
    -------
    submit.sh: submition script for raven
    
    """
    script_content = f"""#!/bin/bash -l
#SBATCH --no-requeue                    
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node={core}            
#SBATCH --job-name={job_name}            
#SBATCH --partition={partition}              
#SBATCH --time={time}                    
#SBATCH -o ./tjob.out.%j                 
#SBATCH -e ./tjob.err.%j                 
#SBATCH --mem={disk_space}    
           
module load parallel/201807
export OMP_NUM_THREADS=1

which python
{run_command}
"""

    # Writing to a file
    with open(f"submit.sh", "w") as f:
        f.write(script_content)

    return 0




parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter)

parser.add_argument('-data', '--data',
                    default=None,
                    type=str,
                    help="The string of dataset! \
                        \n-d data.xyz"
                    )

parser.add_argument('-code', '--code',
                    default=None,
                    type=str,
                    help="The code for DFTB calculation! \
                        \ne.g. -c calc_dftb.py "
                    )

parser.add_argument('-method', '--method',
                    default=None,
                    nargs='*',
                    # type=str,
                    help='Run DFTB with either DFTB or GFN-xTb \
                        \n-m static(xtb1 or xtb2) 5(kpts_dens)'
                    )

parser.add_argument('-Node', '--node',
                    default=None,
                    type=int,
                    help='How many Node do you want to use \
                        \nNote: which means how many sub jobs do you want to submit \
                        \n-N 3')

parser.add_argument('-core', '--core',
                    default=None,
                    type=int,
                    help='The parallel cobre number on each node!\
                        \n-c 40')
parser.add_argument('-mpcdf', '--mpcdf',
                    default=None,
                    type=str,
                    help='local, raven, or cobra')

parall_args = parser.parse_args()

from pathlib import Path
p = Path.cwd()
calculator_code = parall_args.code
tag = parall_args.method


submit_dict = {
    'time':"00:29:00",
    'disk_space':90000,
    'job_name':"DFTB",
}
if parall_args.mpcdf == 'raven':
    submit_dict['partition'] = "general"
    submit_dict['core'] = 72

elif parall_args.mpcdf == 'cobra':
    submit_dict['partition'] = "middle"
    submit_dict['core'] = 40
elif parall_args.mpcdf == 'viper':
    submit_dict['partition'] = 'general'
    submit_dict['core'] = 128

elif parall_args.mpcdf == 'local':
    print(f"Run gnuparallel on {parall_args}!")

else:
    print("Please set -mpcdf is raven, cobra, or viper!")

if parall_args.mpcdf == 'cobra' or parall_args.mpcdf == 'raven' or parall_args.mpcdf == 'viper':

    if parall_args.node is not None:
        dataset_names = seperate_atoms(parall_args.data, int(parall_args.node), 'node')
        print("dataset_names:", dataset_names)
        for n, dataset in enumerate(dataset_names):
            folder_dir = p / f"node_{n}"
            dataset_dir = p / dataset
            print(dataset_dir)
            os.chdir(folder_dir)

            prep_gnuparallel(str(dataset_dir), int(parall_args.core), calculator_code, tag, submit_dict)

            os.chdir(p)
    else:
        print("Please set -node to >= 1")

elif parall_args.mpcdf == 'local':
    prep_gnuparallel(str(parall_args.data), int(parall_args.core), calculator_code, tag)
    

