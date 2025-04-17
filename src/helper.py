import numpy as np
import pyscipopt as scp
import torch


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
def normalize_features(features : str):
    maxs = torch.max(features, 0)[0]# maximos de cada feature (el [0] es porque max retorna una tupla de (valores,indices))
    mins = torch.min(features, 0)[0]# minimos de cada feature (el [0] es porque min retorna una tupla de (valores,indices))
    diff = maxs - mins#Rango de cada feature
    for ks in range(diff.shape[0]):#recorre cada feature
        if diff[ks] == 0:#Si la feature es constante, se asume que su rango es 1
            diff[ks] = 1
    #se normaliza para que las features esten entre 0 y 1
    return torch.clamp((features - mins)/ diff, 1e-5, 1)#se limita para que esten entre 1e-5 y 1

def normalize_linear_sum(linear_sum : str):
    linear_sum = linear_sum.split()
    sum_neg_coeff = 0
    new_sum = []
    for i in range(len(linear_sum)//2):
        coeff = int(linear_sum[2*i])
        var = linear_sum[2*i+1]
        if coeff<0:
            sum_neg_coeff+=coeff #acumula coeff negativos
            coeff = -coeff #Convierte los coeffs a positivos
            var = f"~{var}" if var[0]!="~" else var[1:] #niega el literal (~x -> x, x -> ~x)
        new_sum.append("+" + str(coeff))
        new_sum.append(var)
    return " ".join(new_sum), sum_neg_coeff

def normalize_obj(obj_func : str):
    obj = obj_func.split(":")[0]
    linear_sum,_ = normalize_linear_sum(obj_func.split(":")[1])
    return f"{obj}: {linear_sum};\n"

def normalize_contraint(contraint:str):
    if ">=" in contraint:
        sign = ">="
    elif "<=" in contraint:
        sign = "<="
    elif ">" in contraint:
        sign = ">"
    elif "<" in contraint:
        sign = "<"
    elif "=" in contraint:
        sign = "="
    else:
        raise Exception(f"La constraint no es >=, <=, >, < ni = : {contraint}")
    right_side = int(contraint.split(sign)[1].split()[0])
    linear_sum, sum_neg_coeff = normalize_linear_sum(contraint.split(sign)[0])
    return f"{linear_sum} {sign} {right_side-sum_neg_coeff};\n"

def normalize_instance(ins_name:str):
    instance =ins_name.split("/")[-1]
    temp_file = "/tmp/"+instance
    with open(ins_name,"r") as f:
        lines = f.readlines()
        with open(temp_file,"w") as wf:
            for i in range(len(lines)):
                if lines[i][0]=="*":
                    continue
                if lines[i].startswith("min") or lines[i].startswith("max"):
                    wf.write(normalize_obj(lines[i]))
                    continue
                wf.write(normalize_contraint(lines[i]))
    return temp_file

def get_bipartite_graph(ins_name : str):    
    ins_name = normalize_instance(ins_name)
    m = scp.Model()
    m.hideOutput(True)
    m.readProblem(ins_name)
    cons = m.getConss() #lista de constraints
    #Saca de la lista de constraint aquellas que no tienen coeficientes
    new_cons = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        if len(coeff) == 0:
            continue
        new_cons.append(c)
    cons = new_cons
    ncons = len(cons) #Cantidad de constraints no vacias
    #Ordena las constraints segun el numero de variables que tiene (en caso de empate se ordenan por nombre de constraint)
    cons_map = [[x, len(m.getValsLinear(x))] for x in cons]
    cons_map = sorted(cons_map, key=lambda x: [x[1], str(x[0])])
    cons = [x[0] for x in cons_map]

    n_lits = 0
    l_map = {} #diccionario tipo {l_name: l_index}
    l_nodes = []
    #literal embedding
    # [position] name: meaning
    # [0] obj: normalized coefficient of literal in the objective function
    # [1] l_coeff: average coefficient of the literal in all constraints
    # [2] Nl_coeff: degree of literal node in the bipartite representation (en cuantas restricciones aparece el LITERAL)
    # [3] max_coeff: maximum value among all coefficients of the literal
    # [4] min_coeff: minimum value among all coefficients of the literal
    # [5] Nv_coeff: degree of variable node in the bipartite representation (en cuantas restricciones aparece la VARIABLE)
    mvars = m.getVars() #lista de variables
    mvars.sort(key=lambda v: v.name) #ordena variables por nombre
    for i in range(len(mvars)): #recorre cada variable
        l_nodes.append([0,0,0,0,1e+20,0])
        l_nodes.append([0,0,0,0,1e+20,0])
    
    l_map = {} # diccionario tipo {nombreVaribale: indice}
    for indx, v in enumerate(mvars):
        l_map[v.name] = 2*indx
        l_map[v.name+"_neg"] = 2*indx + 1
    c_nodes = []
    #constraint embedding
    # [position] name: meaning
    # [0] c_coeff: average of all coefficients in the constraint
    # [1] Nc_coeff: degree of constraint nodes in the bipartite representation
    # [2] rhs: right-hand-side value of the constraint
    # [3] sense: the sense of the constraint
    indices_spr = [[], []] # Coordenadas de arcos [constraintIndexes, variableIndexes]
    values_spr = [] # Peso que habra en la matriz de adjacencia (siempre seran 1)

    for cind, c in enumerate(cons): #itera sobre las constraints
        coeff = m.getValsLinear(c) #diccionario de coeficientes {nombreVariable: coeficiente}
        rhs = m.getRhs(c)
        lhs = m.getLhs(c)
        sense = 0 # Constraint del tipo variables <= rhs

        if rhs == lhs:
            sense = 2 # Constraint de igualdad ==
        elif rhs >= 1e+20:
            sense = 1 # Constraint tipo lhs <= variables (variables >= lhs)
            rhs = lhs

        summation = 0
        for lit in coeff: #Recorre los nombres de las variables
            if coeff[lit] != 0: #Si el coeficienet no es 0 añade el arco que conecta la restriccion con el literal
                l_indx = l_map[lit] #obtiene el indice del literal
                indices_spr[0].append(cind) #indice de la constraint
                indices_spr[1].append(l_indx) #indice de la variable
                values_spr.append(1) 
            
                l_nodes[l_indx][1] += coeff[lit] / ncons #Añade el coeficiente normalizado (segun la restriccion) al embedding de la variable
                l_nodes[l_indx][2] += 1 #Agrega 1 al contador de restricciones en la que participa el literal
                
                l_nodes[l_indx][3] = max(l_nodes[l_indx][3], coeff[lit]) # Actualiza el maximo coeficiente de la variable
                l_nodes[l_indx][4] = min(l_nodes[l_indx][4], coeff[lit]) # Actualiza e minimio coeficiente de la variable
                l_nodes[l_indx][5] += 1 #Agrega 1 al contador de restricciones en la que participa la variable
                neg_lit = lit+"_neg" if lit[-4:] != "_neg" else lit[:-4]
                l_nodes[l_map[neg_lit]][5] = l_nodes[l_indx][5] #Actualiza el contador tambien en su negacion

                summation += coeff[lit] #Acumula todos los coeficientes de la restriccion
        llc = max(len(coeff), 1) #numero de coeficientes en la restriccion (asume que almenos hay 1)
        c_nodes.append([summation / llc, llc, rhs, sense]) #agrega los embeddings de la restriccion
    obj = m.getObjective()
    is_optimization = len(obj.terms)!=0
    if is_optimization:
        obj_node = [0, 0, 0, 0] # nodo (restriccion) que representa la funcion objetivo
        for e in obj: #terminos en la funcion objetivo
            vnm = e.vartuple[0].name #nombre de la variable
            v = obj[e] #coeficiente de la variable
            #v_indx = v_map[vnm] #indice de la variable
            if v != 0: # Si el coeficiente no es 0 añade el arco que une la variable a la restriccion (func. objetivo)
                if v>0:
                    lit = vnm
                else:
                    lit = vnm+"_neg"
                l_indx = l_map[lit] #obtiene el indice del literal
                indices_spr[0].append(ncons) #Indice de la restriccion (func. objetivo)
                indices_spr[1].append(l_indx) #Indice del literal
                values_spr.append(1)
                l_nodes[l_indx][0] = v #añade el coeficiente de la variable en la funcion objetivo al embedding de la variable

                obj_node[0] += v #acumula los coeficientes de todas las variables
                obj_node[1] += 1 #cuanta cuantos coeficientes hay (cuantas varibles)
        
        obj_node[0] /= obj_node[1] #calcula la media de los coeficientes en la restriccion (func. objetivo)
        c_nodes.append(obj_node)
        ncons+=1
    
    l_nodes = torch.as_tensor(l_nodes, dtype=torch.float32)#.to(device)
    c_nodes = torch.as_tensor(c_nodes, dtype=torch.float32)#.to(device)
    l_nodes = normalize_features(l_nodes)
    c_nodes = normalize_features(c_nodes)

    A = torch.sparse_coo_tensor(indices_spr, values_spr, (ncons, len(mvars)*2))
    return A, l_map, l_nodes, c_nodes
