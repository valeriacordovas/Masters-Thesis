"""
 GUIDETree.py  

    Author: Valeria Cordova
    This code is based on Anson Wong's ModelTree function,
    available at https://github.com/ankonzoid/LearningX
"""
import numpy as np
from copy import deepcopy
from graphviz import Digraph
import scipy.stats
import pandas as pd
from itertools import combinations
import math


class GUIDETree(object):

    def __init__(self, model, max_depth=5, min_samples_leaf=10):

        self.model = model
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def get_params(self, deep=True):
        return {
            "model": self.model.get_params() if deep else self.model,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def __repr__(self):
        class_name = self.__class__.__name__
        return "{}({})".format(class_name, ', '.join([ "{}={}".format(k,v) for k, v in self.get_params(deep=False).items() ]))

    # ======================
    # Fit
    # ======================
    def fit(self, X, y, var_types, verbose=False):

        # Settings
        model = self.model
        min_samples_leaf = self.min_samples_leaf
        max_depth = self.max_depth
        var_idx = list(range(np.size(X,1)))
        
        global labels
        labels = list(np.unique(y))

        if verbose:
            print(" max_depth={}, min_samples_leaf={}...".format(max_depth, min_samples_leaf))

        def _build_tree(X, y):

            global index_node_global

            def _create_node(X, y, Xog, var_idx, var_types, depth, container):
                loss_node, model_node = _fit_model(X, y, model, var_types, 0)
                node = {"name": "node",
                        "index": container["index_node_global"],
                        "loss": loss_node,
                        "model": model_node,
                        "endog": y,
                        "exog": X,
                        "data_og": Xog,
                        "var_idx": var_idx,
                        "var_types": var_types,
                        "n_samples": len(X),
                        "j_feature": None,
                        "type_best": None,
                        "threshold": None,
                        "children": {"left": None, "right": None},
                        "depth": depth}
                container["index_node_global"] += 1
                return node

            # Recursively split node + traverse node until a terminal node is reached
            def _split_node(node, container):

                # Split node and save results
                result = _splitter(node, model,
                                   max_depth=max_depth,
                                   min_samples_leaf=min_samples_leaf)

                # Return leaf if not split
                if not result["did_split"]:
                    node["name"] = "leaf"
                    if verbose:
                        depth_spacing_str = " ".join([" "] * node["depth"])
                        print(" {}*leaf {} @ depth {}: loss={:.6f}, N={}".format(depth_spacing_str, node["index"], node["depth"], node["loss"], result["N"]))
                    return

                # Update node information based on splitting result
                node["j_feature"] = result["j_feature"]
                node["type_best"] = result["type_best"]
                node["threshold"] = result["threshold"]

                # Extract results
                (X_left, y_left), (X_right, y_right) = result["data"]
                (X_left_og), (X_right_og) = result["data_og"]
                model_left, model_right = result["models"]
                types_left, types_right = result["var_types"]
                var_idx_left, var_idx_right = result["var_idx"]

                # Print splitting results
                if verbose:
                    depth_spacing_str = " ".join([" "] * node["depth"])
                    if node["type_best"] == "cat" or node["type_best"] == "bin":
                        print(" {}node {} @ depth {}: loss={:.3f}, j_feature={}, threshold={}, N=({},{})".format(depth_spacing_str, node["index"], node["depth"], node["loss"], node["j_feature"], node["threshold"], len(X_left), len(X_right)))
                    else: 
                        print(" {}node {} @ depth {}: loss={:.3f}, j_feature={}, threshold={:.1f}, N=({},{})".format(depth_spacing_str, node["index"], node["depth"], node["loss"], node["j_feature"], node["threshold"], len(X_left), len(X_right)))
                        
                # Create children nodes
                node["children"]["left"]  = _create_node(X_left, y_left, X_left_og, var_idx_left, types_left, node["depth"]+1, container)
                node["children"]["right"] = _create_node(X_right, y_right, X_right_og, var_idx_right, types_right, node["depth"]+1, container)
                node["children"]["left"]["model"] = model_left
                node["children"]["right"]["model"] = model_right

                # Split children nodes
                _split_node(node["children"]["left"], container)
                _split_node(node["children"]["right"], container)

            container = {"index_node_global": 0}  # node indices
            root = _create_node(X, y, X, var_idx, var_types, 0, container)  # create root node at depth 0
            _split_node(root, container)  # start root node splitting 

            return root

        # Construct tree
        self.tree = _build_tree(X, y)

    # ======================
    # Predict
    # ======================
    def predict(self, X, Xog, type = "prob"):
        assert self.tree is not None
        
        y_pred = np.array(_predict(self.tree, X[0,:], type))
        for x in X[1:,:]:
            y_pred = np.vstack((y_pred, np.array(_predict(self.tree, x, type))))
            
        return y_pred

    # ======================
    # Explain
    # ======================
    def explain(self, X, header):
        assert self.tree is not None
        def _explain(node, x, explanation):
            no_children = node["children"]["left"] is None and \
                          node["children"]["right"] is None
            if no_children:
                return explanation
            else:
                if node["type_best"] == "cat" or node["type_best"] == "bin":
                    if np.isin(x[node["j_feature"]], node["threshold"]):  # x[j] == threshold
                        explanation.append("{} = {:.6f} == {}".format(header[node["j_feature"]], x[node["j_feature"]], node["threshold"]))
                        return _explain(node["children"]["left"], x, explanation)
                    else: # x[j] != threshold
                        explanation.append("{} = {:.6f} != {}".format(header[node["j_feature"]], x[node["j_feature"]], node["threshold"]))
                        return _explain(node["children"]["right"], x, explanation)
                else:
                    if x[node["j_feature"]] <= node["threshold"]:  # x[j] < threshold
                        explanation.append("{} = {:.6f} <= {:.6f}".format(header[node["j_feature"]], x[node["j_feature"]], node["threshold"]))
                        return _explain(node["children"]["left"], x, explanation)
                    else:  # x[j] > threshold
                        explanation.append("{} = {:.6f} > {:.6f}".format(header[node["j_feature"]], x[node["j_feature"]], node["threshold"]))
                        return _explain(node["children"]["right"], x, explanation)

        explanations = [_explain(self.tree, x, []) for x in X]
        return explanations
    
    # ======================
    # Get leaves
    # ======================
    def get_leaves(self):
        assert self.tree is not None
        
        tree = deepcopy(self.tree)
        leaves = _get_leaves(tree)
        return leaves
    
    # =============================
    # Get observations' leaf index
    # =============================
    def get_leaf_idx(self, X):
        assert self.tree is not None
        
        leaves = np.array([_get_leaf_idx(self.tree, x, []) for x in X])
        return leaves
    
    # ======================
    # Prune
    # ======================
    def prune(self):
        assert self.tree is not None
        
        def _prune(tree, alphas = []): 
            tree_copy = deepcopy(tree)

            no_children = tree_copy.tree["children"]["left"] is None and \
                          tree_copy.tree["children"]["right"] is None
                          
            if no_children and tree_copy.tree["index"] == 0:
                # Finish pruning when root node has no children nodes
                return alphas
            else:
                # compute alpha for each subtree                
                aux_alpha = np.array(_get_alphas(tree_copy.tree))
                
                # select minimum alpha for pruning
                alpha = np.min(aux_alpha[:,1])
                                
                n_mins = aux_alpha[aux_alpha[:,1] == alpha,1]   # num. subtrees with same g_t()
                if len(n_mins) == 1:
                    idx_cut = int(aux_alpha[np.where(aux_alpha == alpha)[0], 0])
                elif len(n_mins) > 1:
                    # if more than one subtree with minimum alpha, select the one that prunes less leaves 
                    min_leaves_cut = np.min(aux_alpha[aux_alpha[:,1] == alpha, 2])                                                                                
                    idx_cut = int(np.where(np.all([aux_alpha[:,1] == alpha, aux_alpha[:,2] == min_leaves_cut], axis=0))[0][0]) # select first node index that meets both conditions
                
                # collapse node "idx_cut" into leaf                     
                _cut_tree(tree_copy.tree, idx_cut)
                
                print(" Found alpha = {:.3f} for subtree with {} leaves".format(alpha, _get_nleaves(tree_copy.tree)))  
                                
                sub_alpha = [alpha, tree_copy]
                
                alphas.extend([sub_alpha])
                
                return _prune(sub_alpha[1], alphas)
                        
        def _get_alphas(node, result = []):
            no_children = node["children"]["left"] is None and \
                          node["children"]["right"] is None
            
            if no_children:
                return
            else:
                leaves_cut = _get_nleaves(node)
                result = [[node["index"], _gt(node), leaves_cut]]
                
                g_left = _get_alphas(node["children"]["left"])
                if g_left:
                    result.extend(g_left)
                
                g_right = _get_alphas(node["children"]["right"])
                if g_right:
                    result.extend(g_right)
              
                return result
        
        def _gt(node):
            # Training loss of a subtree Tt - a tree with root at node t
            loss_tree = np.empty((len(node["endog"]),1))
            for i in range(len(loss_tree)):
                loss_tree[i] = _loss_tree(node, node["data_og"][i,:], node["endog"][i], labels)
            rt_t = np.sum(loss_tree)
            
            # Training loss of node t
            rt = node["loss"]
            
            n_leaves = _get_nleaves(node)
            
            gt = (rt - rt_t)/(n_leaves - 1)
            
            return gt
         
        global tree_obj
        tree_obj = deepcopy(self)
        global N_train
        N_train = np.size(tree_obj.tree["data_og"],0)
        global labels
        labels = np.unique(tree_obj.tree["endog"])
        
        print("Cost-complexity pruning starting with alpha = 0 and {} leaves".format(_get_nleaves(tree_obj.tree)))
        return _prune(tree_obj, alphas = [[0, tree_obj]])
        
    # ======================
    # Loss
    # ======================
    def loss(self, X, y):
        labels = np.unique(y)
        loss = np.empty((len(y),1))
        for i in range(len(y)):
            loss[i] = _loss_tree(self.tree, X[i,:], y[i], labels)
        return np.sum(loss)

    # ======================
    # RSS
    # ======================
    def rss(self):
        rss = _rss_tree(self.tree)
        return rss

    # ======================
    # Tree diagram
    # ======================
    def export_graphviz(self, output_filename, feature_names,
                        export_png=True, export_pdf=False):
        """
         Assumes node structure of:
           node["name"]
           node["n_samples"]
           node["children"]["left"]
           node["children"]["right"]
           node["j_feature"]
           node["type_best"]
           node["threshold"]
           node["loss"]
        """
        g = Digraph('g', node_attr={'shape': 'record', 'height': '.1'})

        def build_graphviz_recurse(node, parent_node_index=0, parent_depth=0, edge_label=""):
            # Empty node
            if node is None:
                return

            # Create node
            node_index = node["index"]
            if node["children"]["left"] is None and node["children"]["right"] is None:
                threshold_str = ""
            else:
                if node["type_best"] == "cat" or node["type_best"] == "bin":
                    threshold_str = "{} = {}\\n".format(feature_names[node['j_feature']], node["threshold"])
                else:
                    threshold_str = "{} <= {:.1f}\\n".format(feature_names[node['j_feature']], node["threshold"])

            label_str = "({})\\n {} N = {}\\n loss = {:.3f}".format(node["index"], threshold_str, node["n_samples"], node["loss"])

            # Create node
            nodeshape = "rectangle"
            bordercolor = "black"
            fillcolor = "white"
            fontcolor = "black"
            g.attr('node', label=label_str, shape=nodeshape)
            g.node('node{}'.format(node_index),
                   color=bordercolor, style="filled",
                   fillcolor=fillcolor, fontcolor=fontcolor)

            # Create edge
            if parent_depth > 0:
                g.edge('node{}'.format(parent_node_index),
                       'node{}'.format(node_index), label=edge_label)

            # Traverse child or append leaf value
            build_graphviz_recurse(node["children"]["left"],
                                   parent_node_index=node_index,
                                   parent_depth=parent_depth + 1,
                                   edge_label="")
            build_graphviz_recurse(node["children"]["right"],
                                   parent_node_index=node_index,
                                   parent_depth=parent_depth + 1,
                                   edge_label="")

        # Build graph
        build_graphviz_recurse(self.tree,
                               parent_node_index=0,
                               parent_depth=0,
                               edge_label="")

        # Export pdf
        if export_pdf:
            print("Saving model tree diagram to '{}.pdf'...".format(output_filename))
            g.format = "pdf"
            g.render(filename=output_filename, view=False, cleanup=True)

        # Export png
        if export_png:
            print("Saving model tree diagram to '{}.png'...".format(output_filename))
            g.format = "png"
            g.render(filename=output_filename, view=False, cleanup=True)


# ***********************************
#
# Additional functions
#
# ***********************************

def _splitter(node, model,
              max_depth=5, min_samples_leaf=10):

    # Extract node's data
    X = node["exog"]
    y = node["endog"]
    Xog = node["data_og"]
    depth = node["depth"]
    var_idx = node["var_idx"]
    var_types = node["var_types"]
    N, d = X.shape

    # Set preliminary results
    did_split = False
    loss_best = node["loss"]
    data_best = None
    data_og = Xog
    var_idx_new = None
    models_best = None
    data_types = None
    j_feature_best = None
    type_feature_best = None
    threshold_best = None

    # Perform split if and only if node is not at max depth
    if (depth >= 0) and (depth < max_depth):
        
        ## Estimate full (node) model and get residuals
        resid = _fit_model(X, y, model, var_types, 1)
        
        ## Select split variable (use data without dummy-coded categorical variables)
        split, split_var = _select_split_var(X, y, model, resid, var_types)
        
        # Do not continue if no split is found
        if not split:
            result = {"did_split": did_split,
                      "loss": loss_best,
                      "models": models_best,
                      "data": data_best,
                      "data_og": data_og,
                      "var_idx": var_idx_new,
                      "var_types": data_types,
                      "j_feature": j_feature_best,
                      "type_best": type_feature_best,
                      "threshold": threshold_best,
                      "N": N}
            return result
        
        # Use greedy search to generate threshold search list
        if var_types[split_var] == "cat":
            threshold_search = []
            for i in range(math.trunc((len(set(X[:,split_var])))/2)): # this goes from 0 to [(#categories/2)-1]
                threshold_search.extend(list(combinations(set(X[:,split_var]), i+1)))    # +1 to go from 1 to #categories/2
        else:
            threshold_search = list(set(X[:,split_var]))[:-1]
            
        # Perform threshold split search on split_var
        for threshold in threshold_search:

            # Split data at threshold
            (X_left_og, y_left_og), (X_right_og, y_right_og) = _split_data(var_idx[split_var], threshold, Xog, y, var_types[split_var])
            (X_left, y_left), (X_right, y_right) = _split_data(split_var, threshold, X, y, var_types[split_var])
            N_left, N_right = len(X_left), len(X_right)
                        
            # Splitting conditions
            split_conditions = [N_left >= min_samples_leaf,
                                N_right >= min_samples_leaf]

            # Do not attempt split if split conditions not satisfied
            if not all(split_conditions):
                continue
            
            # Eliminate variables with only one value (constant)
            drop_left  = []
            drop_right = []
            for i in range(d):
                if len(set(X_left[:,i])) == 1 and var_types[i] != "cons":
                    drop_left.append(i)
                if len(set(X_right[:,i])) == 1 and var_types[i] != "cons":
                    drop_right.append(i)
        
            X_left  = np.delete(X_left,  drop_left,  1)
            X_right = np.delete(X_right, drop_right, 1)
            idx_left = np.delete(var_idx, drop_left).tolist()
            idx_right = np.delete(var_idx, drop_right).tolist()
            types_left  = np.delete(var_types, drop_left).tolist()
            types_right = np.delete(var_types, drop_right).tolist()
                        
            # Compute split loss 
            loss_left, model_left   = _fit_model(X_left,  y_left,  model, types_left,  0)
            loss_right, model_right = _fit_model(X_right, y_right, model, types_right, 0)
            
            loss_split = (loss_left + loss_right)
            
            # Update best parameters if loss is lower
            if loss_split < loss_best:
                did_split = True
                loss_best = loss_split
                models_best = [model_left, model_right]
                data_best = [(X_left, y_left), (X_right, y_right)]
                data_og = [(X_left_og), (X_right_og)]
                var_idx_new = [idx_left, idx_right]
                data_types = [types_left, types_right]
                j_feature_best = var_idx[split_var]
                type_feature_best = var_types[split_var]
                threshold_best = threshold

    # Return the best result
    result = {"did_split": did_split,
              "loss": loss_best,
              "models": models_best,
              "data": data_best,
              "data_og": data_og,
              "var_idx": var_idx_new,
              "var_types": data_types,
              "j_feature": j_feature_best,
              "type_best": type_feature_best,
              "threshold": threshold_best,
              "N": N}

    return result

def _fit_model(X, y, model, var_types, resid):
    model_copy = deepcopy(model) 
    # Transform categorical variables to dummies for estimation
    cats = []
    for i in range(np.size(X,1)):
        if var_types[i] == "cat" or var_types[i] == "ord":
            cats.append(i)
    
    Xest = pd.get_dummies(pd.DataFrame(X), columns = cats, drop_first=True).to_numpy()
   
    model_copy.fit(Xest, y, maxiter=10000, cov_type='HC0', disp=0)
    
    if resid == 0:
        try:
            loss = model_copy.loss(Xest, y, labels)
        except Exception:
            loss = float("inf")
            pass
        
        assert loss >= 0.0
        return loss, model_copy
    
    elif resid == 1:
        res = model_copy.residuals()
        return res

def _split_data(j_feature, threshold, X, y, var_type):
    if var_type == "cat":
        idx_left = np.where(np.isin(X[:, j_feature], threshold))[0]
    else:
        idx_left = np.where(X[:, j_feature] <= threshold)[0]
    
    idx_right = np.delete(np.arange(0, len(X)), idx_left)
    assert len(idx_left) + len(idx_right) == len(X)
    return (X[idx_left], y[idx_left]), (X[idx_right], y[idx_right])

def _transform(x, X, var_idx, var_types):
    ## Dummy-code categorical variables for each row of original data
    # var_idx: Node's variables' original index 
    # var_types: Node's variables type
    x = x[var_idx]
    
    cats = []
    for i in range(len(var_types)):
        if var_types[i] == "cat" or var_types[i] == "ord":
            cats.append(var_idx[i])

    # All dummies in original dataset                
    dummy_map = {}
    for i in range(len(cats)):
        dummy_map[cats[i]] = list(set(X[:,cats[i]]))[1:]           # [1:] excludes first category
    c = [f'{k}_{x}' for k, v in dummy_map.items() for x in v]      # get dummy-encoded names
    
    # Drop observations if their value corresponds to first category in categorical or ordered variable
    x_drop = []
    idx_drop = []
    for i in range(len(x)):
        if np.isin(var_types[i], ["cat", "ord"]): 
            if x[i] == np.min(X[:,var_idx[i]]):
                x_drop.append(i) 
                idx_drop.append(i)    
    
    xnew = np.delete(x, x_drop)
    idx_new = list(np.delete(np.array(var_idx), idx_drop))  
    
    # Recover original index of categorical variables not dropped
    cats_new = []
    for i in range(len(xnew)):
        if np.isin(var_types[var_idx.index(idx_new[i])], ["cat", "ord"]):
            cats_new.append(idx_new[i])
    
            
    # Transform categorical variables to dummies (first columns already excluded in xnew)
    df_encoded = pd.get_dummies(pd.DataFrame(np.reshape(xnew,(1,len(xnew))), columns = idx_new), columns = cats_new, drop_first=False)
    
    vals = df_encoded.columns.union(c)
    
    xnew = df_encoded.reindex(vals, axis=1, fill_value=0) 
    
    # Order categorical variables in ascending order to match pd.get_dummies from original estimation
    non_cat = []
    for i in idx_new:
        if not np.isin(i,cats_new):
            non_cat.append(i)
    
    cats_dcode = list(set(xnew.columns) - set(non_cat))
    
    reordered = non_cat[:]
    for col in c:
        if np.isin(col, cats_dcode):
            reordered.append(col)
    
    xfin = xnew[reordered].to_numpy()
    
    return xfin

def _select_split_var(X, y, model, resid, var_types):
    # Perform curvature and interaction tests for each variable
    N, d = X.shape

    idx_res = np.empty(shape=(N,1))
    X = np.c_[X,idx_res]
    
    for i in range(N):
        if resid[i] > 0:
            X[i,-1] = 1
        else:
            X[i,-1] = 0
    
    result = np.empty((1,3)) # cols = var1, var2 (interaction), p-val
    lowest_pval = [10, np.nan]
    for var in range(d):
        if var_types[var] == "cons":
            continue
        if len(set(X[:,var])) == 1:
            continue
        
        ## Curvature test
        if var_types[var] == "cont":
            Oij = np.zeros(shape=(2,4))
            Eij = np.zeros(shape=(2,4))
            aux = np.zeros(shape=(2,4))
        else:
            Oij = np.zeros(shape=(2,len(set(X[:,var]))))
            Eij = np.zeros(shape=(2,len(set(X[:,var]))))
            aux = np.zeros(shape=(2,len(set(X[:,var]))))
        
        if np.isin(var_types[var], ["cat", "ord"]):
            vals = list(set(X[:,var]))
            for i in range(0,2): # residuals split
                for j in range(len(set(X[:,var]))): # categorical variable levels (must be integers)
                    Oij[i,j] = len(X[(X[:,var] == vals[j]) & (X[:,-1] == i)])
                    Eij[i,j] = ((len(X[(X[:,var] == vals[j])]))*(len(X[(X[:,-1] == i)])))/N
                    
                    aux[i,j] = ((Oij[i,j]-Eij[i,j])**2)/(Eij[i,j])
                
        elif var_types[var] == "bin":
            for i in range(0,2): # residuals split
                for j in set(X[:,var]): # binary variable levels (0,1)
                    j = int(j)
                    Oij[i,j] = len(X[(X[:,var] == j) & (X[:,-1] == i)])
                    Eij[i,j] = ((len(X[(X[:,var] == j)]))*(len(X[(X[:,-1] == i)])))/N
                    
                    aux[i,j] = ((Oij[i,j]-Eij[i,j])**2)/(Eij[i,j])
        
        elif var_types[var] == "cont":
            qnt = np.quantile(X[:,var], [0.25, 0.5, 0.75])
            for i in range(0,2): # residuals split
                for j in range(0,4): # continuous variable quantiles
                    if j == 0:
                        Oij[i,j] = len(X[(X[:,var] <= qnt[j]) & (X[:,-1] == i)])
                        Eij[i,j] = ((len(X[(X[:,var] <= qnt[j])]))*(len(X[(X[:,-1] == i)])))/N
                    elif (j > 0) and (j < 3):
                        Oij[i,j] = len(X[(X[:,var] > qnt[j-1]) & (X[:,var] <= qnt[j]) & (X[:,-1] == i)])
                        Eij[i,j] = ((len(X[(X[:,var] > qnt[j-1]) & (X[:,var] <= qnt[j])]))*(len(X[(X[:,-1] == i)])))/N
                    elif j == 3:
                        Oij[i,j] = len(X[(X[:,var] > qnt[j-1]) & (X[:,-1] == i)])
                        Eij[i,j] = ((len(X[(X[:,var] > qnt[j-1])]))*(len(X[(X[:,-1] == i)])))/N
                    
                    aux[i,j] = ((Oij[i,j]-Eij[i,j])**2)/(Eij[i,j])
                    
        mask = np.all(aux == 0, axis=0) | np.all(np.isnan(aux), axis=0) # remove zero or nan columns      
        aux = aux[:,~mask]
        X2 = np.sum(aux)
        df = aux.shape[1]-1 # this will be (at most) len(set(X[:,var]))-1 (if categorical) or 4-1 (if continuous)
        p_val = scipy.stats.chi2.sf(X2,df)
        
        if len(result) == 1:
            result[0,:] = [var, np.nan, p_val]
        else:
            result = np.vstack([result, [var, np.nan, p_val]])
        
        ## Interaction tests
        if var <= d-2:
            # start from next variable to avoid repeating interactions
            for var2 in range(var+1,d):
                if var == var2:
                    continue
                
                if np.isin(var_types[var], ["cat", "ord", "bin"]):
                    if np.isin(var_types[var2], ["cat", "ord", "bin"]):
                        Oij = np.zeros(shape=(2,len(set(X[:,var]))*len(set(X[:,var2]))))
                        Eij = np.zeros(shape=(2,len(set(X[:,var]))*len(set(X[:,var2]))))
                        aux = np.zeros(shape=(2,len(set(X[:,var]))*len(set(X[:,var2]))))
                        
                        vals1 = list(set(X[:,var]))
                        vals2 = list(set(X[:,var2]))
                        adj = len(set(X[:,var2]))
                        for i in range(2):
                            for j in range(len(set(X[:,var]))):
                                for k in range(len(set(X[:,var2]))):
                                    Oij[i,(j*adj)+k] = len(X[(X[:,var] == vals1[j]) & (X[:,var2] == vals2[k]) & (X[:,-1] == i)])
                                    Eij[i,(j*adj)+k] = ((len(X[(X[:,var] == vals1[j]) & (X[:,var2] == vals2[k])]))*(len(X[(X[:,-1] == i)])))/N
                            
                            for j in range(np.size(aux,1)):
                                aux[i,j] = ((Oij[i,j]-Eij[i,j])**2)/(Eij[i,j])
                                
                    elif var_types[var2] == "cont":
                        Oij = np.zeros(shape=(2,2*len(set(X[:,var]))))
                        Eij = np.zeros(shape=(2,2*len(set(X[:,var]))))
                        aux = np.zeros(shape=(2,2*len(set(X[:,var]))))
                        
                        median = np.quantile(X[:,var2], [0.5])[0]
                        vals = list(set(X[:,var]))
                        adj = len(set(X[:,var]))
                        for i in range(2):
                            for j in range(len(set(X[:,var]))):
                                Oij[i,j] = len(X[(X[:,var2] <= median) & (X[:,var] == vals[j]) & (X[:,-1] == i)])
                                Eij[i,j] = ((len(X[(X[:,var2] <= median) & (X[:,var] == vals[j])]))*(len(X[(X[:,-1] == i)])))/N
                                
                                Oij[i,j+adj] = len(X[(X[:,var2] > median) & (X[:,var] == vals[j]) & (X[:,-1] == i)])
                                Eij[i,j+adj] = ((len(X[(X[:,var2] > median) & (X[:,var] == vals[j])]))*(len(X[(X[:,-1] == i)])))/N
                                
                                aux[i,j] = ((Oij[i,j]-Eij[i,j])**2)/(Eij[i,j])
                                aux[i,j+adj] = ((Oij[i,j+adj]-Eij[i,j+adj])**2)/(Eij[i,j+adj])
                
                elif var_types[var] == "cont":
                    if np.isin(var_types[var2], ["cat", "ord", "bin"]):
                        Oij = np.zeros(shape=(2,2*len(set(X[:,var2]))))
                        Eij = np.zeros(shape=(2,2*len(set(X[:,var2]))))
                        aux = np.zeros(shape=(2,2*len(set(X[:,var2]))))
                        
                        median = np.quantile(X[:,var], [0.5])[0]
                        vals = list(set(X[:,var2]))
                        adj = len(set(X[:,var2]))
                        for i in range(2):
                            for j in range(len(set(X[:,var2]))):
                                Oij[i,j] = len(X[(X[:,var] <= median) & (X[:,var2] == vals[j]) & (X[:,-1] == i)])
                                Eij[i,j] = ((len(X[(X[:,var] <= median) & (X[:,var2] == vals[j])]))*(len(X[(X[:,-1] == i)])))/N
                                
                                Oij[i,j+adj] = len(X[(X[:,var] > median) & (X[:,var2] == vals[j]) & (X[:,-1] == i)])
                                Eij[i,j+adj] = ((len(X[(X[:,var] > median) & (X[:,var2] == vals[j])]))*(len(X[(X[:,-1] == i)])))/N
                                
                                aux[i,j] = ((Oij[i,j]-Eij[i,j])**2)/(Eij[i,j])
                                aux[i,j+adj] = ((Oij[i,j+adj]-Eij[i,j+adj])**2)/(Eij[i,j+adj])
                                
                    elif var_types[var2] == "cont":
                        Oij = np.zeros(shape=(2,4))
                        Eij = np.zeros(shape=(2,4))
                        aux = np.zeros(shape=(2,4))
                        
                        median1 = np.quantile(X[:,var], [0.5])[0]
                        median2 = np.quantile(X[:,var2], [0.5])[0]
                        for i in range(2):
                            Oij[i,0] = len(X[(X[:,var] <= median1) & (X[:,var2] <= median2) & (X[:,-1] == i)])
                            Eij[i,0] = ((len(X[(X[:,var] <= median1) & (X[:,var2] <= median2)]))*(len(X[(X[:,-1] == i)])))/N
                            
                            Oij[i,1] = len(X[(X[:,var] <= median1) & (X[:,var2] > median2) & (X[:,-1] == i)])
                            Eij[i,1] = ((len(X[(X[:,var] <= median1) & (X[:,var2] > median2)]))*(len(X[(X[:,-1] == i)])))/N
                            
                            Oij[i,2] = len(X[(X[:,var] > median1) & (X[:,var2] <= median2) & (X[:,-1] == i)])
                            Eij[i,2] = ((len(X[(X[:,var] > median1) & (X[:,var2] <= median2)]))*(len(X[(X[:,-1] == i)])))/N
                            
                            Oij[i,3] = len(X[(X[:,var] > median1) & (X[:,var2] > median2) & (X[:,-1] == i)])
                            Eij[i,3] = ((len(X[(X[:,var] > median1) & (X[:,var2] > median2)]))*(len(X[(X[:,-1] == i)])))/N
                            
                            for j in range(4):
                                aux[i,j] = ((Oij[i,j]-Eij[i,j])**2)/(Eij[i,j])
                            
                mask = np.all(aux == 0, axis=0) | np.all(np.isnan(aux), axis=0) # remove zero or nan columns      
                aux = aux[:,~mask]
                X2 = np.sum(aux)
                df = aux.shape[1]-1 
                p_val = scipy.stats.chi2.sf(X2,df)
                
                result = np.vstack([result, [var, var2, p_val]])
    
    conf_lvl = 0.1 
    
    mask = np.isnan(result[:,2]) # remove nan p-values (row)      
    result = result[~mask,:]
        
    lowest_pval[0] = np.min(result[:,2])
    if lowest_pval[0] > conf_lvl:
        split = False
    else:
        split = True
        if np.isnan(result[np.where(result == np.min(result[:,2]))[0],1][0]):
            lowest_pval[1] = int(result[np.where(result == np.min(result[:,2]))[0],0][0])    
        else:
            var1 = int(result[np.where(result == np.min(result[:,2]))[0],0][0])
            var2 = int(result[np.where(result == np.min(result[:,2]))[0],1][0])
            
            if np.isin(var_types[var1], ["cat", "ord", "bin"]) and np.isin(var_types[var2], ["cat", "ord", "bin"]):
                p_val1 = result[np.where(np.logical_and(result[:,0] == var1, np.isnan(result[:,1])))[0],2][0]
                p_val2 = result[np.where(np.logical_and(result[:,0] == var2, np.isnan(result[:,1])))[0],2][0]
                if p_val1 <= p_val2:
                    lowest_pval[1] = var1
                else:
                    lowest_pval[1] = var2
                    
            elif np.isin(var_types[var1], ["cat", "ord", "bin"]) and var_types[var2] == "cont":
                lowest_pval[1] = var1
                
            elif var_types[var1] == "cont" and np.isin(var_types[var2], ["cat", "ord", "bin"]):
                lowest_pval[1] = var2
                
            elif var_types[var1] == "cont" and var_types[var2] == "cont":
                # Temporarily split on variables' means, select the one that reduces loss the most
                mean1 = np.mean(X[:,var1])
                mean2 = np.mean(X[:,var2])
                
                (X_left_1, y_left_1), (X_right_1, y_right_1) = _split_data(var1, mean1, X[:,:-1], y, var_types[var1])
                (X_left_2, y_left_2), (X_right_2, y_right_2) = _split_data(var2, mean2, X[:,:-1], y, var_types[var2])
                
                drop_left_1  = []
                drop_right_1 = []
                drop_left_2  = []
                drop_right_2 = []
                for i in range(d):
                    if len(set(X_left_1[:,i])) == 1 and var_types[i] != "cons":
                        drop_left_1.append(i)
                    if len(set(X_right_1[:,i])) == 1 and var_types[i] != "cons":
                        drop_right_1.append(i)
                    if len(set(X_left_2[:,i])) == 1 and var_types[i] != "cons":
                        drop_left_2.append(i)
                    if len(set(X_right_2[:,i])) == 1 and var_types[i] != "cons":
                        drop_right_2.append(i)
            
                X_left_1  = np.delete(X_left_1,  drop_left_1,  1)
                X_right_1 = np.delete(X_right_1, drop_right_1, 1)
                types_left_1  = np.delete(var_types, drop_left_1).tolist()
                types_right_1 = np.delete(var_types, drop_right_1).tolist()
                
                X_left_2  = np.delete(X_left_2,  drop_left_2,  1)
                X_right_2 = np.delete(X_right_2, drop_right_2, 1)
                types_left_2  = np.delete(var_types, drop_left_2).tolist()
                types_right_2 = np.delete(var_types, drop_right_2).tolist()
                
                # Compute split losses
                loss_left_1,_  = _fit_model(X_left_1,  y_left_1,  model, types_left_1,  0)
                loss_right_1,_ = _fit_model(X_right_1, y_right_1, model, types_right_1, 0)
                loss_1 = (loss_left_1 + loss_right_1)
                
                loss_left_2,_  = _fit_model(X_left_2,  y_left_2,  model, types_left_2,  0)
                loss_right_2,_ = _fit_model(X_right_2, y_right_2, model, types_right_2, 0)
                loss_2 = (loss_left_2 + loss_right_2)
                
                if loss_1 <= loss_2:
                    lowest_pval[1] = var1
                else:
                    lowest_pval[1] = var2
               
    return split, lowest_pval[1]

def _get_leaves(node):
    global leaves
    try: 
        leaves
    except NameError:
        leaves = []
    
    no_children = node["children"]["left"] is None and \
                  node["children"]["right"] is None    
    if no_children:
        leaves.append(node)
        return leaves
    else:
        _get_leaves(node["children"]["left"])
        _get_leaves(node["children"]["right"])
      
        return leaves

def _get_leaf_idx(node, x, leaf):
    no_children = node["children"]["left"] is None and \
                  node["children"]["right"] is None
    if no_children:
        leaf.append(node["index"])
        return leaf
    else:
        if node["type_best"] == "cat" or node["type_best"] == "bin":
            if np.isin(x[node["j_feature"]], node["threshold"]):  # x[j] == threshold
                return _get_leaf_idx(node["children"]["left"], x, leaf)
            else: # x[j] != threshold
                return _get_leaf_idx(node["children"]["right"], x, leaf)
        else:
            if x[node["j_feature"]] <= node["threshold"]:  # x[j] < threshold
                return _get_leaf_idx(node["children"]["left"], x, leaf)
            else:  # x[j] > threshold
                return _get_leaf_idx(node["children"]["right"], x, leaf)

def _get_nleaves(tree):
    # Initialize count
    try: 
        leaves_count
    except NameError:
        leaves_count = 0   
    
    no_children = tree["children"]["left"] is None and \
                          tree["children"]["right"] is None
    if no_children:
        leaves_count += 1
        return leaves_count
    else:
        return _get_nleaves(tree["children"]["left"]) + _get_nleaves(tree["children"]["right"])
    
def _predict(node, x, type = "prob", transform = True):
    no_children = node["children"]["left"] is None and \
                  node["children"]["right"] is None
    if no_children:
        if transform:
            x = _transform(x, node["data_og"], node["var_idx"], node["var_types"])
        if type == "prob":
            y_pred_x = node["model"].predict(x, type)[0]
        else:
            y_pred_x = node["model"].predict(x, type)
        return y_pred_x
    else:
        if node["type_best"] == "cat" or node["type_best"] == "bin":
            if np.isin(x[node["j_feature"]], node["threshold"]):  # x[j] == threshold
                return _predict(node["children"]["left"], x, type)
            else:  # x[j] != threshold
                return _predict(node["children"]["right"], x, type)
        else:
            if x[node["j_feature"]] <= node["threshold"]:  # x[j] < threshold
                return _predict(node["children"]["left"], x, type)
            else:  # x[j] > threshold
                return _predict(node["children"]["right"], x, type)

def _loss_tree(node, x, y, labels):
    no_children = node["children"]["left"] is None and \
                          node["children"]["right"] is None
    if no_children:
        x_t = _transform(x, node["data_og"], node["var_idx"], node["var_types"])
        loss = node["model"].loss(x_t, y, labels)
        return loss
    else:
        if node["type_best"] == "cat" or node["type_best"] == "bin":
            if np.isin(x[node["j_feature"]], node["threshold"]):  # x[j] == threshold
                return _loss_tree(node["children"]["left"], x, y, labels)
            else:  # x[j] != threshold
                return _loss_tree(node["children"]["right"], x, y, labels)
        else:
            if x[node["j_feature"]] <= node["threshold"]:  # x[j] < threshold
                return _loss_tree(node["children"]["left"], x, y, labels)
            else:  # x[j] > threshold
                return _loss_tree(node["children"]["right"], x, y, labels)

def _rss_tree(node, rss = 0):  
    no_children = node["children"]["left"] is None and \
                          node["children"]["right"] is None
    
    if no_children:
        rss += node["model"].rss()
        return rss
    else:
        return _rss_tree(node["children"]["left"]) + _rss_tree(node["children"]["right"])

def _cut_tree(tree, node):
    no_children = tree["children"]["left"] is None and \
                          tree["children"]["right"] is None
    
    if tree["index"] == node:
        tree["children"]["left"]  = None 
        tree["children"]["right"] = None
        tree["name"] = "leaf"
        
    elif not no_children:
        tree["children"]["left"]  = _cut_tree(tree["children"]["left"], node)
        tree["children"]["right"] = _cut_tree(tree["children"]["right"], node)
        
    return tree                                           