import os
import torch
import torch.nn.functional as F
from get_bipartite_graph import get_bipartite_graph
import shutil
import pickle
import sys
from GCN import GraphDatasetTriOutput, BackboneAndValuesPredictor
import time
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import random
import pyscipopt as scp
def merge_index(l_map):
    new_map = {}
    for literal in l_map:
        if l_map[literal]%2==0:
            new_map[literal] = l_map[literal]//2
    return new_map
def get_trust_region_constraints_tri_output(model,instances_path,instance_name,parameters):
    P,RP = parameters
    instance_input = f"{instances_path}/{instance_name}"
    #get bipartite graph as input
    A, l_map, l_nodes, c_nodes = get_bipartite_graph(instance_input)
    v_map = merge_index(l_map)
    constraint_features, edge_indices, edge_features, variable_features = GraphDatasetTriOutput.get_graph_components(A, l_nodes, c_nodes)
    
    #prediction
    with torch.no_grad():
        BD = model(
            constraint_features.to(DEVICE),
            edge_indices.to(DEVICE),
            edge_features.to(DEVICE),
            variable_features.to(DEVICE),
        )
        pred = F.softmax(BD,dim=1).cpu().squeeze()
    N = len(v_map)
    k = int(N*P)
    Delta = int(N*P*(1-RP))
    selected = torch.topk(pred[:,:2].max(dim=1)[0],k)[1]
    pred[:,1] = pred[:,1]>=pred[:,0]
    pred[:,0] = pred[:,1]<pred[:,0]

    k_0 = pred[:,0][selected].sum().item()
    k_1 = pred[:,1][selected].sum().item()
    
    print(f'instance: {instance_name}, fix {k_0} 0s and fix {k_1} 1s, Delta {Delta}. ')
    
    index_to_var_name = {v_map[var_name]:var_name for var_name in v_map}
    scp_model = scp.Model()
    scp_model.hideOutput()
    scp_model.readProblem(instance_input)
    scp_var_map = {}  # Map to store original variables and their complements
    # Step 1: Duplicate binary variables with complements
    for var in scp_model.getVars():
        scp_var_map[var.name] = var
    constraints = []
    deltas = []
    #iterate over indexes of selected
    for i in selected:
        tar_var = index_to_var_name[i.item()] #target variable
        delta_var = scp_model.addVar(name=f"delta_{tar_var}", vtype="B")
        if pred[i,0].item():
            constraints.append(scp_var_map[tar_var]<=delta_var)
        elif pred[i,1].item():
            constraints.append(1-scp_var_map[tar_var]<=delta_var)
        else:
            raise Exception("No se ha predicho un valor valido")
        deltas.append(delta_var)
    if len(deltas)>0:
        constraints.append(sum(deltas) <= Delta )
    return constraints, scp_model
def add_contraints(scp_model,instance_name,target_path,contraints):
    dst = f"{target_path}/{instance_name}"
    for c in contraints:
        scp_model.addCons(c)
    scp_model.writeProblem(dst)

def main():
    # dataset_path: carpeta con las representaciones de las instancias (archivos .sol y .bg)
    # workdir: Carpeta donde almacenar los logs y los modelos entrenados
    # partition: Archivo que contine la lista de las intancias particionada (cuales usar para train y cuales para test)
    
    if len(sys.argv) != 7:
        print("Usage: create_trust_region.py <dataset> <partition_path> <log_path> <files_limit> <P> <RP>")
        print("Example: create_trust_region.py /src/datasets/pbo16 /src/wkdir/pbo16/exp1/0.pkl /src/wdir/pbo16/exp1/logs 60 0.1 0.9")
        sys.exit(1)
    dataset_path = sys.argv[1]
    partition_path = sys.argv[2]
    log_path = sys.argv[3]
    files_limit = int(sys.argv[4])
    parameters = float(sys.argv[5]),float(sys.argv[6])
    P,RP = parameters #percentage, retrieval precision
    print(f"Execution: {' '.join(sys.argv)}")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.isdir(f'{log_path}'):
        print(f"No se encuentra logpath: {log_path}")
        sys.exit(1)
    bestmodel_path = f"{log_path}/model_best.pth"
    instances_path = f"{dataset_path}/instance"

    #lectura de particiones y creacion de dataloaders
    with open(partition_path, 'rb') as f:
        partitions = pickle.load(f)
    
    model = BackboneAndValuesPredictor().to(DEVICE)
    state = torch.load(bestmodel_path)
    model.load_state_dict(state)
    model.eval()
    for partition_name in partitions["tests"]:
        ps_instances_path = f'{log_path}/test/{partition_name}/predict_and_search_P_RP_{P:6f}_{RP:6f}'
        if not os.path.exists(ps_instances_path):
            os.makedirs(ps_instances_path)
        else:
            print(f"Ya existe {ps_instances_path}")
            continue
        orig_instances_path = f'{log_path}/test/{partition_name}/orig'
        instances_list = partitions["tests"][partition_name]
        if files_limit>0:
            instances_list = random.sample(instances_list,min(files_limit,len(instances_list)))
        if not os.path.exists(orig_instances_path):
            os.makedirs(orig_instances_path)
            for instance_name in instances_list:
                src = f"{instances_path}/{instance_name}"
                dst = f"{orig_instances_path}/{instance_name}"
                shutil.copyfile(src, dst)
        instances_list = os.listdir(orig_instances_path)
        for instance_name in instances_list:
            #Crea copia con trust region
            print(f"creando copia de {instance_name}")
            begin=time.time()
            constraints,scp_model = get_trust_region_constraints_tri_output(model,instances_path,instance_name,parameters)
            if len(constraints)>0:
                add_contraints(scp_model,instance_name,ps_instances_path,constraints)
                process_time = time.time()-begin
                print(f"copia de {instance_name} creada en {str(process_time)}")
            else:
                process_time = time.time()-begin
                print(f"No se ha creado copia ya que no se predijo trust region")
            #Crea copia sin trust region
        

if __name__ == "__main__":
    main()