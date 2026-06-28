import os, math, sys
import ase.io


def prep_gnuparallel(dataset_infile:str, num_parall, calculator_code:str, tag:str):
    """
    Be parallelized !!!

    Parameters:
    ----------
    dataset_file: str
        path to the validation dataset file
    num_parall: int
        number of parallel processes
    tag: str
        tag for the output folders
    """

    # 1. Divide input file into num_parall folders
    # initialize information
    dataset = ase.io.read(dataset_infile, ':')
    dataset_file = dataset_infile.strip().split('/')[-1]
    dataset_file_output = dataset_file.replace('.xyz', f'_{tag}.xyz')
    num_atoms = len(dataset)
    

    # seperate val_dataset into num_parall parts
    size_subs = math.ceil(num_atoms / num_parall)

    subdataset_names = [f"{tag}_{i}/tmp.{dataset_file.split('.')[0]}.xyz" for i in range(num_parall)]
    subdataset_gap_names = [f"{tag}_{i}/tmp.{dataset_file.split('.')[0]}_dftb.xyz" for i in range(num_parall)]
    print(f"subdataset_names: {'  '.join(subdataset_names)}")
    print(f"subdataset_gap_names: {'  '.join(subdataset_gap_names)}")


    # write subdataset
    cmd_file = open("cmd.lst", 'w')
    for n in range(num_parall):
        start_index = n * size_subs
        end_index = (n + 1) * size_subs
        subdataset = dataset[start_index: end_index]

        subfolder_name = f"{tag}_{n}"
        if not os.path.exists(subfolder_name):
            os.system(f"mkdir {subfolder_name}")
            os.system(f"cp {calculator_code} {subfolder_name}")
            if tag.lower() != "xtb":
                os.system(f"cp input/*.skf {subfolder_name}")
        else:
            os.system(f"cp {calculator_code} {subfolder_name}")
            if tag.lower() == "dftb":
                os.system(f"cp *.skf {subfolder_name}")
        subdataset_name = subdataset_names[n]  
        ase.io.write(subdataset_name, subdataset)


        # write cmd file to cmd.lst, here i run calculator with python
        cmd_run = f"python  {calculator_code} -sp tmp.{dataset_file.split('.')[0]}.xyz {tag}"

        cmd_file.write(f"cd {subfolder_name} ; {cmd_run} >> output\n")
    cmd_file.close()

    print("Seperate validation dataset into " + str(num_parall) + " parts for parallel calculation.")


    # 2. Run GAP on each part
    os.system(f"parallel --delay 0.2 --joblog task.log --progress --resume -j {num_parall} < cmd.lst")


    # 3. Collect data after finished
    os.system(f"cat {' '.join(subdataset_gap_names)} > {dataset_file_output}")
    os.system("mv task.log task_backup.log")


    output_dataset = ase.io.read(dataset_file_output, ':')
    if len(output_dataset) == len(dataset):
        os.system(f"rm -r {tag}_*")
    else:
        print(f"WARNING: len(ref_dataset):{len(dataset)} != len(gap_dataset):{len(output_dataset)}")


dataset_file, num_parall = sys.argv[1:]

# I need to specifiy the calculator code
calculator_code = "calc_dftb.py"
tag = "static"

prep_gnuparallel(dataset_file, int(num_parall), calculator_code, tag)
