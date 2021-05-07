"""
 OModel.py  (author: Valeria Cordova)
 Wrapper for Statsmodels' Ordered Logit/Probit Model
"""
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
from sklearn.utils.validation import check_is_fitted
import pdb

class OModel:

    def __init__(self, model_class, classes, distr = 'logit'):
        self.model_class = model_class
        self.classes = classes
        self.distr = distr

    def fit(self, X, y, maxiter=1000, cov_type='HC0', method = 'bfgs', skip_hessian=True, disp=1):
        """
         Available methods are: (statsmodels)
            - 'newton' for Newton-Raphson (slower), 'nm' for Nelder-Mead
            - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
            - 'lbfgs' for limited-memory BFGS with optional box constraints
            - 'powell' for modified Powell's method
            - 'cg' for conjugate gradient
            - 'ncg' for Newton-conjugate gradient
            - 'basinhopping' for global basin-hopping solver
            - 'minimize' for generic wrapper of scipy minimize (BFGS by default)
        """
        self.model = OrderedModel(y, X, distr= self.distr)
        self.result = self.model.fit(maxiter=1000, cov_type=cov_type, method=method, skip_hessian = skip_hessian, disp=disp)
        return self.result
    
    def summary(self):
        # Check is fit had been called
        check_is_fitted(self, 'model')
        return self.result.summary()

    def predict(self, X, type = "prob"):
        if type == "prob":
            pred = self.result.predict(X)
        elif type == "choice":
            pred = np.asarray(self.result.predict(X)).argmax(1) + np.min(self.classes)
        elif type == "lin":
            pred = self.result.predict(X, which = "linpred")
        else:
            print("Invalid type given. Type = ['prob','choice','lin']")

        return pred
    
    def residuals(self):
        # Residual Probability
        return self.result.resid_prob
    
    def rss(self):
        sqresid = np.square(self.result.resid_prob)
        return np.sum(sqresid)
        
    def loss(self, X, y, labels = []):
        # All-Threshold Loss (Rennie & Srebro, 2005)
        # Note: y need to be transformed to (consecutive) numeric categories
        # labels: optional if y does not contain all the categories of the data
        linpred = self.result.predict(X, which = "linpred")       
        thresh = self.model.transform_threshold_params(self.result.params)
        
        if not len(labels):
            labels = np.unique(y)
        try:
            np.size(y,1)
        except:
            y = y.reshape(len(y),1)
            
         
        loss = np.empty((len(y),1))
        for n in range(len(y)):
            f = []
            if thresh[y[n,0]-np.min(labels)] < linpred[n] <= thresh[y[n,0]-np.min(labels)+1]:
                loss[n] = 0
            else:
                for i in range(len(thresh[1:-1])):
                    if i + np.min(labels) < y[n,0]:
                        f.append(np.log(1 + np.exp(-(linpred[n] - thresh[i+1]))))
                    else:
                        f.append(np.log(1 + np.exp(-(thresh[i+1] - linpred[n]))))
                
                loss[n] = np.sum(f)
        
        return np.sum(loss)    
