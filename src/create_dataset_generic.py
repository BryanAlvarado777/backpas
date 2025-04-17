import os.path
import pickle
from multiprocessing import Process, Queue, set_start_method
import numpy as np
import argparse
from get_bipartite_graph import get_bipartite_graph
from helper_pas import get_a_new2
import pandas as pd
import sys
here = os.path.dirname(__file__)

def get_backbone_target(backbone_path: str, l_map:dict):
    target = [0]*len(l_map)
    complete_backbone = False
    with open(backbone_path, 'r') as f:
        #iterate over lines
        for line in f:
            line = line.strip()
            parts =  line.split()
            if parts[0] != "b":
                continue
            literal = parts[1]
            if literal[0] == "-":
                literal = literal[1:]+"_complement"
            elif literal == "0":
                complete_backbone = True
                break
            target[l_map[literal]] = 1
    if not complete_backbone:
        raise Exception("Backbone not complete")
    target = np.array(target,dtype=np.float32)
    return target

def transoform_vmap_to_lmap(v_map:dict):
    l_map = {key: 2 * idx for key, idx in v_map.items()}
    l_map.update({key + "_complement": 2 * idx + 1 for key, idx in v_map.items()})
    return l_map
def collect(instance_path:str, backbone_path:str, filename:str ,ml_dataset_path :str):
    if not filename:
        return

    instance_filepath = os.path.join(instance_path,filename)
    backbone_filepath = os.path.join(backbone_path,filename+".backbone")
    try:
        #get bipartite graph , binary variables' indices
        #bipartite_graph = get_bipartite_graph(instance_filepath)
        A,v_map,v_nodes,c_nodes,b_vars =get_a_new2(instance_filepath)
        bipartite_graph=(A,v_map,v_nodes,c_nodes)
        
        literals_map = transoform_vmap_to_lmap(v_map)
        backbone_target = get_backbone_target(backbone_filepath,literals_map)
        data = {
            "X": bipartite_graph,
            "y": backbone_target,
        }
        pickle.dump(data, open(os.path.join(ml_dataset_path, filename+'.pkl'), 'wb'))
    except Exception as e:
        print(f"Error procesando archivo {filename}: {e}")
    
    





if __name__ == '__main__':
    set_start_method('spawn')
    if len(sys.argv) != 2:
        print("Usage: create_dataset.py <dataset>")
        print("Example: create_dataset.py /src/datasets/pbo16")
        sys.exit(1)
    dataset_path = sys.argv[1]
    
    instance_path=os.path.join(dataset_path,"instance")
    backbone_path=os.path.join(dataset_path,"backbone")
    ml_dataset_path=os.path.join(dataset_path,"ml_dataset")
    
    os.makedirs(ml_dataset_path,exist_ok=True)
    
    instance_filenames = os.listdir(instance_path)
    backbone_filenames = os.listdir(backbone_path)
    backbone_filenames = [f.replace(".backbone","") for f in backbone_filenames]
    filenames = list(set(instance_filenames).intersection(set(backbone_filenames)))
    q = []
    # add ins
    for filename in filenames:
        #Añade los que faltan
        if not os.path.exists(os.path.join(ml_dataset_path,filename+'.pkl')):
            q.append(filename)
    l_q = len(q)
    for i in range(l_q):
        print(f"Procesando {i}/{l_q}: {q[i]}")
        #collect(instance_path,backbone_path,q[i],ml_dataset_path)
        p = Process(target=collect,args=(instance_path,backbone_path,q[i],ml_dataset_path))
        p.start()
        p.join(60*10)
        if p.is_alive():
            print(f"Timeout {{q[i]}}, terminate()")
            p.terminate()
            p.join()
        
    print('done')


