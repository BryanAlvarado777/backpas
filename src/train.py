import os
import sys
import pickle
from GCN import GraphDatasetTriOutput, MilpBackboneAndValuesPredictor,BackboneAndValuesPredictor
import torch_geometric
import torch
import torch.nn.functional as F
import time
import numpy as np
from torcheval.metrics.functional import multiclass_accuracy,multiclass_f1_score,multiclass_auroc, multiclass_auprc,multiclass_precision, multiclass_recall , retrieval_precision, multiclass_confusion_matrix
import logging
#import warnings
#warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

class MetricAggregator():
    def __init__(self, metrics_func_dict:dict, multi_metrics_func_dict:dict = None):
        """
            metrics_func_dict: {name_of_the_metric: function_to_compute_the_metric}
            multi_metrics_func_dict: {general_name_of_the_metric: (function_to_compute_the_metric,{specific_name_of_the_metric:index_to_acces_the_metric})}
        """
        self.metrics_func_dict = metrics_func_dict
        self.multi_metrics_func_dict = multi_metrics_func_dict
        self.metric_names = [k for k in self.metrics_func_dict]
        self.metric_names.sort()
        self.multi_metric_names = [k for k in self.multi_metrics_func_dict]
        self.multi_metric_names.sort()
        self.multi_metric_specific_names = []
        for general_name in self.multi_metric_names:
            for specific_name in self.multi_metrics_func_dict[general_name][1]:
                self.multi_metric_specific_names.append(general_name+"_"+specific_name)
        self.reset()
    def get_metric_names(self):
        return self.metric_names+self.multi_metric_specific_names
    def reset(self):
        self.values = [[] for _ in self.metric_names] + [[] for _ in self.multi_metric_specific_names]
    def update(self,pred,target):
        for index, metric in enumerate(self.metric_names):
            self.values[index].append(self.metrics_func_dict[metric](pred,target).item())
        index = len(self.metric_names)
        for metric in self.multi_metric_names:
            metric_func, indexes_to_access_metric = self.multi_metrics_func_dict[metric]
            value = metric_func(pred,target)
            for index_to_access in indexes_to_access_metric.values():
                self.values[index].append(value[index_to_access].item())
                index+=1
    def aggregate(self):
        metrics = np.array(self.values)
        return metrics.mean(axis=1).tolist()
def create_precision_at_p(p:float):
    def precision_at_p(pred,target):
        #pred is a tensor with shape (n_variables,3)
        pred = F.softmax(pred,dim=1)
        #the last dimension is the probability of each category (backbone with value 0, backbone with value 1, non-backbone)
        #calculate max between backbone with value 0 and backbone with value 1
        predicted_is_backbone = pred[:,:2].max(dim=1)[0]
        #predicted_is_backbone = pred[:,0]+pred[:,1]
        #select first 2 columns of the prediction
        is_prediction_correct = (pred[:,:2].argmax(dim=1)==target).int()
        k = int(predicted_is_backbone.nelement()*p)
        #top_k_index = torch.topk(predicted_is_backbone,k)[1]
        #return is_prediction_correct[top_k_index].sum().float()/k
        return retrieval_precision(predicted_is_backbone,is_prediction_correct,k)
    return precision_at_p
def create_dataloader_from_names(names:list,dataset_path:str,batch_size:int,shuffle:bool=True):
    files = [ os.path.join(dataset_path,name+".pkl") for name in names]
    graph_dataset = GraphDatasetTriOutput(files)
    return torch_geometric.loader.DataLoader(graph_dataset, batch_size=batch_size, shuffle=shuffle)
def one_epoch(model, data_loader, metric_aggregator, device, batch_accumulation,optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()
    mean_loss = []
    metric_aggregator.reset()
    n_samples_processed = 0
    backward_passes = 0
    n_train_samples = len(data_loader)
    with torch.set_grad_enabled(optimizer is not None):
        for step, batch in enumerate(data_loader):
            batch = batch.to(device)
            pred = model(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
            #iterate each instance
            start_index = 0
            loss = torch.zeros(1,device=device)
            for i in range(batch.ntvars.shape[0]):
                end_index = start_index+batch.ntvars[i]
                # instance slices
                instance_pred = pred[start_index:end_index]
                instance_target = batch.y[start_index:end_index]
                # loss de una instancia
                #instance_loss = F.binary_cross_entropy(instance_pred,instance_target,reduction="mean")
                instance_loss = F.cross_entropy(instance_pred,instance_target,reduction="mean")
                loss += instance_loss
                # stats
                mean_loss.append(instance_loss.item())
                metric_aggregator.update(instance_pred,instance_target)
                start_index = end_index
            n_samples_processed += batch.num_graphs
            if optimizer is not None:
                (loss/batch.num_graphs).backward()
                if (n_samples_processed >= (backward_passes+1) * batch_accumulation) or n_samples_processed==n_train_samples:
                    optimizer.step()
                    optimizer.zero_grad()
                    backward_passes+=1
    
    return np.mean(mean_loss), metric_aggregator.aggregate()
def full_train(model,optimizer,epochs,best_model_path,last_model_path,add_to_log, metric_aggregator, device,batch_accumulation, train_loader,valid_loader=None,scheduler=None):
    best_loss = float("inf")
    for epoch in range(epochs):
        print(epoch)
        begin=time.time()
        loss, metrics = one_epoch(model, train_loader, metric_aggregator, device,batch_accumulation, optimizer=optimizer)
        epoch_time = time.time()-begin
        add_to_log(epoch,epoch_time,"train",[loss]+metrics)
        begin=time.time()
        if valid_loader:
            begin=time.time()
            loss, metrics = one_epoch(model, valid_loader, metric_aggregator, device, batch_accumulation, optimizer=None)
            if scheduler != None:
                scheduler.step(loss)
            epoch_time = time.time()-begin
            add_to_log(epoch,epoch_time,"valid",[loss]+metrics)
            begin=time.time()
        
        if loss<best_loss:
            best_loss = loss
            torch.save(model.state_dict(),best_model_path)
        torch.save(model.state_dict(), last_model_path)
def main():
    # dataset_path: carpeta con las representaciones de las instancias (archivos .sol y .bg)
    # workdir: Carpeta donde almacenar los logs y los modelos entrenados
    # partition: Archivo que contine la lista de las intancias particionada (cuales usar para train y cuales para test)
    TEST_ONLY = False
    if len(sys.argv) != 4:
        print("Usage: train.py <dataset> <partition_path> <log_path>")
        print("Example: train.py /src/datasets/pbo16 /src/wkdir/pbo16/exp1/0.pkl /src/wdir/pbo16/exp1/logs")
        #print command executed
        print("Execution: ", ' '.join(sys.argv))
        sys.exit(1)
    dataset_path = sys.argv[1]
    partition_path = sys.argv[2]
    log_path = sys.argv[3]
    print(f"Execution: {' '.join(sys.argv)}")
    if not os.path.isdir(f'{log_path}'):
        os.makedirs(f'{log_path}')
    if TEST_ONLY:
        logfile = open(f"{log_path}/metrics.log", 'a')
    else:
        logfile = open(f"{log_path}/metrics.log", 'w')
    def add_to_log(epoch,time,partition,metrics):
        logfile.write(",".join(str(val) for val in ([epoch,time,partition]+metrics))+"\n")
        logfile.flush()
    bestmodel_path = f"{log_path}/model_best.pth"
    lastmodel_path = f"{log_path}/model_last.pth"
    
    # hiperparametros del modelo (y el entrenamiento)
    BATCH_ACCUMULATION = 32 
    BATCH_SIZE = 1 #Para el entrenamiento
    LEARNING_RATE = 0.001
    EPOCHS = 200 #numero de epocas de entrenamiento
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #lectura de particiones y creacion de dataloaders
    with open(partition_path, 'rb') as f:
        partitions = pickle.load(f)
    train_loader = create_dataloader_from_names(partitions["train"],dataset_path,BATCH_SIZE,shuffle=True)
    if "valid" in partitions:
        valid_loader = create_dataloader_from_names(partitions["valid"],dataset_path,BATCH_SIZE,shuffle=False)
    else:
        valid_loader = None

    #Metricas
    all_metrics_functions = {
        "accuracy_macro":lambda input,target: multiclass_accuracy(input,target,average="macro",num_classes=3),
        "accuracy_micro":lambda input,target: multiclass_accuracy(input,target,average="micro",num_classes=3),
        "f1_score_macro":lambda input,target: multiclass_f1_score(input,target,average="macro",num_classes=3),
        "f1_score_micro":lambda input,target: multiclass_f1_score(input,target,average="micro",num_classes=3),
        "precision_macro":lambda input,target: multiclass_precision(input,target,average="macro",num_classes=3),
        "precision_micro":lambda input,target: multiclass_precision(input,target,average="micro",num_classes=3),
        #"recall_macro":lambda input,target: multiclass_recall(input,target,average="macro",num_classes=3),
        #"recall_micro":lambda input,target: multiclass_recall(input,target,average="micro",num_classes=3),
    }
    classes_indexes= {"V0":0,"V1":1,"NB":2}
    # Create the confusion matrix mapping using the provided indices
    confusion_matrix_mapping = {
        f"cm_{row_class}_{col_class}": classes_indexes[row_class] * len(classes_indexes) + classes_indexes[col_class]
        for row_class in classes_indexes
        for col_class in classes_indexes
    }

    all_multi_metric_functions = {
        "multiclass_accuracy":(lambda input,target: multiclass_accuracy(input,target,average=None,num_classes=3),classes_indexes),
        "multiclass_precision":(lambda input,target: multiclass_precision(input,target,average=None,num_classes=3),classes_indexes),
        "multiclass_recall":(lambda input,target: multiclass_recall(input,target,average=None,num_classes=3),classes_indexes),
        "multiclass_f1_score":(lambda input,target: multiclass_f1_score(input,target,average=None,num_classes=3),classes_indexes),
        "confusion_matrix":(lambda input,target: multiclass_confusion_matrix(input,target,num_classes=3,normalize="all").flatten(),confusion_matrix_mapping),
    }
    p = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for p_value in p:
        all_metrics_functions[f"retrieval_precision_{p_value:.2f}"] = create_precision_at_p(p_value)
    metric_aggregator = MetricAggregator(all_metrics_functions,all_multi_metric_functions)
    add_to_log("epoch","time","partition",["loss"]+metric_aggregator.get_metric_names())
    #Entrenamiento
    model = BackboneAndValuesPredictor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = None#torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    if not TEST_ONLY:
        full_train(model,optimizer,EPOCHS,bestmodel_path,lastmodel_path,add_to_log,metric_aggregator,DEVICE,BATCH_ACCUMULATION,train_loader,valid_loader,scheduler)
    #Testing
    state = torch.load(bestmodel_path)
    model.load_state_dict(state)
    partitions["tests"]["best_valid"]=partitions["valid"]
    for partition_name in partitions["tests"]:
        #check if all files exists
        test_partition_exists = True
        for name in partitions["tests"][partition_name]:
            if not os.path.exists(os.path.join(dataset_path,name+".pkl")):
                test_partition_exists = False
        if not test_partition_exists:
            print(f"Test partition {partition_name} not found. (Some files are missing)")
            continue
        test_loader = create_dataloader_from_names(partitions["tests"][partition_name],dataset_path,BATCH_SIZE,shuffle=False)
        begin=time.time()
        loss, metrics = one_epoch(model, test_loader, metric_aggregator, DEVICE, BATCH_ACCUMULATION, optimizer=None)
        epoch_time = time.time()-begin
        add_to_log("-1",epoch_time,f"test_{partition_name}",[loss]+metrics)
    logfile.close()

if __name__ == "__main__":
    main()
