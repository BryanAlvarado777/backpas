import os
import pandas as pd
import pickle as pkl
def partition_dataframe(df_param, N):
    """
    Partitions a DataFrame into N parts based on a particular column ordered in ascending order.
    The last partition will contain more rows if the total number of rows is not divisible by N.

    Parameters:
    df (pd.DataFrame): The DataFrame to partition.
    column (str): The column to order by.
    N (int): The number of partitions.

    Returns:
    list: A list of DataFrames, each representing a partition.
    """
    
    # Calculate the size of each partition
    partition_size = len(df_param) // N
    
    # Create the partitions
    partitions = []
    for i in range(N):
        if i < N - 1:
            partitions.append(df_param.iloc[i * partition_size: (i + 1) * partition_size])
        else:
            partitions.append(df_param.iloc[i * partition_size:])
    
    return partitions
def partition_dataframe_randomly(df_param,N):
    df_shuffled = df_param.sample(frac=1).reset_index(drop=True)
    return partition_dataframe(df_shuffled,N)
def partition_dataframe_sorted(df_param,N,sort_column):
    df_sorted = df_param.sort_values(by=sort_column).reset_index(drop=True)
    return partition_dataframe(df_sorted,N)
def get_available_instances(dataset,size_limit=None):
    here = os.path.dirname(__file__)
    bg_path = here+f"/../dataset/{dataset}/input"
    with open(here+"/secret_test_instances.txt","r") as secret_file:
        secret_test_instances = [line.strip() for line in secret_file.readlines()] 
    instances = [(bg[:-3],os.path.getsize(bg_path+"/"+bg)) for bg in os.listdir(bg_path) if bg[:-3] not in secret_test_instances]
    instances = pd.DataFrame(instances,columns=["instance","size"])
    if size_limit!=None:
        return instances[instances["size"]<size_limit]
    return instances
def create_crossval(experiment_path,instances,N=5,name_column="instance"):
    partitions_folder = f"{experiment_path}/partitions"
    os.makedirs(partitions_folder, exist_ok=True)
    partitions = partition_dataframe_randomly(instances,N)
    exp_readme = "Particiones tipo crossval\n"
    infold_train_ratio = 0.8
    for i in range(len(partitions)):
        exp_readme+=f"Pariticion {i}\n"
        train = []
        valid = []
        exp_readme+=f"\tTrain partitions: \n"
        for j,partition in enumerate(partitions[:i]):
            aux_train=partition[name_column].to_list()
            aux_train_cut = int(len(aux_train)*infold_train_ratio)
            train+=aux_train[:aux_train_cut]
            valid+=aux_train[aux_train_cut:]
            exp_readme+=f"\t\tpartition {j} train: {len(aux_train[:aux_train_cut])} instances\n"
            exp_readme+=f"\t\tpartition {j} valid: {len(aux_train[aux_train_cut:])} instances\n"
        for j,partition in enumerate(partitions[i+1:]):
            aux_train=partition[name_column].to_list()
            aux_train_cut = int(len(aux_train)*infold_train_ratio)
            train+=aux_train[:aux_train_cut]
            valid+=aux_train[aux_train_cut:]
            exp_readme+=f"\t\tpartition {j+i+1} train: {len(aux_train[:aux_train_cut])} instances\n"
            exp_readme+=f"\t\tpartition {j+i+1} valid: {len(aux_train[aux_train_cut:])} instances\n"
        tests = {}
        tests[f"partition {i+1}"]= partitions[i][name_column].to_list()
        exp_readme+=f"\tTests partitions:\n"
        exp_readme+=f"\t\tpartition {i}: {len(tests[f'partition {i+1}'])} instances\n"
        with open(f'{partitions_folder}/{i}.pkl', 'wb') as handle:
            pkl.dump({
                "train":train,
                "valid":valid,
                "tests":tests
            }, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open(f'{partitions_folder}/readme.txt', 'w') as readme:
        readme.write(exp_readme)
def create_train_valid_test(experiment_path,instances,train_size=0.6, test_size=0.2, name_column = "instance"):
    partitions_folder = f"{experiment_path}/partitions"
    os.makedirs(partitions_folder, exist_ok=True)
    
    partitions = partition_dataframe_randomly(instances,1)
    exp_readme = "Particiones tipo train valid test\n"
    valid_size = 1-(train_size+test_size)
    assert valid_size>0
    full = partitions[0][name_column].to_list()
    train_valid_cut = int(len(full)*train_size)
    valid_test_cut = int(len(full)*(train_size+valid_size))
    train_partiion = full[:train_valid_cut]
    valid_partiion = full[train_valid_cut:valid_test_cut]
    test_partition = full[valid_test_cut:]
    exp_readme = "Particiones tipo train valid test split\n"
    exp_readme+=f"\tTrain partition: {len(train_partiion)} instances\n"
    exp_readme+=f"\tValid partition: {len(valid_partiion)} instances\n"
    exp_readme+=f"\tTest partition: {len(test_partition)} instances\n"
    assert len(train_partiion)!=0
    assert len(valid_partiion)!=0
    assert len(test_partition)!=0
    tests = {}
    tests[f"0"] = test_partition
    with open(f'{partitions_folder}/0.pkl', 'wb') as handle:
        pkl.dump({
            "train":train_partiion,
            "valid":valid_partiion,
            "tests":tests
        }, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open(f'{partitions_folder}/readme.txt', 'w') as readme:
        readme.write(exp_readme)
def create_train_valid_test_from_lists(experiment_path, train, valid, test):
    partitions_folder = f"{experiment_path}/partitions"
    os.makedirs(partitions_folder, exist_ok=True)
    exp_readme = "Particiones tipo train valid test\n"
    exp_readme+=f"\tTrain partition: {len(train)} instances\n"
    exp_readme+=f"\tValid partition: {len(valid)} instances\n"
    exp_readme+=f"\tTest partition: {len(test)} instances\n"
    assert len(train)!=0
    assert len(valid)!=0
    assert len(test)!=0
    tests = {}
    tests[f"0"] = test
    with open(f'{partitions_folder}/0.pkl', 'wb') as handle:
        pkl.dump({
            "train":train,
            "valid":valid,
            "tests":tests
        }, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open(f'{partitions_folder}/readme.txt', 'w') as readme:
        readme.write(exp_readme)
def create_train_valid_multiple_test_from_lists(experiment_path, train, valid, tests):
    partitions_folder = f"{experiment_path}/partitions"
    os.makedirs(partitions_folder, exist_ok=True)
    exp_readme = "Particiones tipo train valid test\n"
    exp_readme+=f"\tTrain partition: {len(train)} instances\n"
    exp_readme+=f"\tValid partition: {len(valid)} instances\n"
    exp_readme+=f"\tNumber of tests partition: {len(tests)} instances\n"
    for key,val in tests.items():
        exp_readme+=f"\t\tTest partition {key}: {len(val)} instances\n"
    assert len(train)!=0
    assert len(valid)!=0
    with open(f'{partitions_folder}/0.pkl', 'wb') as handle:
        pkl.dump({
            "train":train,
            "valid":valid,
            "tests":tests
        }, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open(f'{partitions_folder}/readme.txt', 'w') as readme:
        readme.write(exp_readme)