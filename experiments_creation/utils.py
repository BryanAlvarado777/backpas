import os
import pickle as pkl
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