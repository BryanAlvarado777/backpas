import os
import numpy as np
from utils import create_train_valid_multiple_test_from_lists
here = os.path.dirname(__file__)
dataset = "MVC-final-ml"
experiment_path = here+f"/../wkdir/{dataset}/full"
print("experiment path",experiment_path)
list_instances = os.listdir(here+f"/../dataset/{dataset}/instance")
backbones = os.listdir(here+f"/../dataset/{dataset}/backbone")
backbones = [backbone.replace(".backbone","") for backbone in backbones]
train_instances = [instance for instance in list_instances if instance.startswith("train_")]
val_instances = [instance for instance in list_instances if instance.startswith("val_")]
test_instances = [instance for instance in list_instances if instance.startswith("test_")]
#remove instances from train and val if they are dont have backbone
train = list(set(train_instances).intersection(set(backbones)))
valid = list(set(val_instances).intersection(set(backbones)))
#split test instances into easy, medium and hard
test = {}
test["easy"] = [instance for instance in test_instances if instance.startswith("test_easy_")]
test["medium"] =  [instance for instance in test_instances if instance.startswith("test_medium_")]
test["hard"] = [instance for instance in test_instances if instance.startswith("test_hard_")]

print("train",len(train))
print("valid",len(valid))
print("test",len(test))
for key in test.keys():
    print("\t",key,len(test[key]))


create_train_valid_multiple_test_from_lists(experiment_path,train,valid,test)


