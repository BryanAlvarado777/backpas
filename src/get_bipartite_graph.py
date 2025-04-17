import pyscipopt as scp
import torch
def normalize_model(model,add_complement_constraints=False):
    """
    Normalize the SCIP model such that all coefficients in constraints are non-negative.
    This may involve duplicating binary variables and adjusting constraints accordingly.
    
    Args:
        model (pyscipopt.scip.Model): The SCIP model to be normalized.
    """
    var_map = {}  # Map to store original variables and their complements
    complement_constraints = {}
    used_complement_vars = set()  # Set to store complement variables that are used in constraints
    # Step 1: Duplicate binary variables with complements
    for var in model.getVars():
        if var.vtype() == "BINARY":  # Check if the variable is binary
            var_map[var.name] = var  # Store the original variable
            # Create a complement variable
            complement_var = model.addVar(name=f"{var.name}_complement", vtype="B")
            var_map[complement_var.name] = complement_var
            complement_constraints[complement_var.name] = (var + complement_var == 1)
    # Step 2: Update constraints
    for cons in model.getConss():
        vals_linear = model.getValsLinear(cons)  # Dictionary of variables and coefficients
        lhs = model.getLhs(cons)  # Left-hand side of the constraint
        rhs = model.getRhs(cons)  # Right-hand side of the constraint
        # Create new expressions for the updated constraint
        new_coefs = {}
        offset = 0
        for var_name, coef in vals_linear.items():

            if coef < 0:  # Handle negative coefficients
                complement_var = var_map[f"{var_name}_complement"]
                new_coefs[complement_var.name] = -coef
                offset -= coef
                used_complement_vars.add(complement_var.name)
            else:
                new_coefs[var_name] = coef
        
        # Remove the old constraint
        model.delCons(cons)

        # Add the new constraint with normalized coefficients
        lin_exp = sum(new_coefs.get(var_name) * var_map.get(var_name) for var_name in new_coefs.keys())
        if lhs!=-model.infinity():
            assert lhs+offset>=-model.infinity()
            model.addCons(
                lin_exp >= (lhs+offset)
            )
        if rhs!=model.infinity():
            assert rhs+offset<=model.infinity()
            model.addCons(
                lin_exp <= (rhs+offset)
            )
     # Step 3: Update the objective function
    obj_exp = model.getObjective()
    new_obj_exp = {}
    obj_offset = 0
    for term, coef in obj_exp.terms.items():
        assert len(term.vartuple) == 1 #verifica que el termino sea un monomio (con 1 sola variable)
        var_name = term.vartuple[0].name
        if coef < 0:
            complement_var = var_map[f"{var_name}_complement"]
            new_obj_exp[complement_var.name] = -coef
            obj_offset += coef
            used_complement_vars.add(complement_var.name)
        else:
            new_obj_exp[var_name] = coef
    obj_lin_exp = sum(new_obj_exp.get(var_name) * var_map.get(var_name) for var_name in new_obj_exp.keys())
    model.setObjective(
        obj_lin_exp,
        sense = model.getObjectiveSense(),
        clear = True
    )
    model.addObjoffset(obj_offset, solutions=False)
    if add_complement_constraints:
        for used_complement_var in used_complement_vars:
            model.addCons(complement_constraints[used_complement_var])
def normalize_features(features : str):
    maxs = torch.max(features, 0)[0]# maximos de cada feature (el [0] es porque max retorna una tupla de (valores,indices))
    mins = torch.min(features, 0)[0]# minimos de cada feature (el [0] es porque min retorna una tupla de (valores,indices))
    diff = maxs - mins#Rango de cada feature
    for ks in range(diff.shape[0]):#recorre cada feature
        if diff[ks] == 0:#Si la feature es constante, se asume que su rango es 1
            diff[ks] = 1
    #se normaliza para que las features esten entre 0 y 1
    return torch.clamp((features - mins)/ diff, 1e-5, 1)#se limita para que esten entre 1e-5 y 1

def get_bipartite_graph(ins_name : str):    
    m = scp.Model()
    m.hideOutput(True)
    m.readProblem(ins_name)
    normalize_model(m)
    
    
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
        l_nodes.append([0,0,0,0,m.infinity(),0])
    
    l_map = {} # diccionario tipo {nombreVaribale: indice}
    indx = 0
    for v in mvars:
        if v.name.endswith("_complement"):
            continue
        l_map[v.name] = 2*indx
        l_map[v.name+"_complement"] = 2*indx + 1
        indx+=1
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
        assert rhs!=m.infinity() or lhs!=-m.infinity() #al menos uno de los dos debe ser finito
        #this assumes that the constraint is of the form lhs <= expr <= rhs with only one of lhs or rhs being finite or both being equal
        if lhs == rhs:
            sense = 0 # exp = rhs
        elif lhs != -m.infinity():
            sense = 1 # exp >= lhs
            rhs = lhs
        elif rhs != m.infinity():
            sense = -1 # exp <= rhs

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
                neg_lit = lit+"_complement" if not lit.endswith("_complement") else lit[:-11]
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
                    lit = vnm+"_complement"
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