import numpy as np

# hyperparameters
C = 1
Lambda = 30
B = 1
option = 1


class olsf:
    def __init__(self):
        self.C = C
        self.Lambda = Lambda
        self.B =  B
        self.option = option
        self.weights = None
        self.wts_init_flag = False

             
    def set_classifier(self, first_example):
        self.weights = np.zeros(len(first_example))
        
        
    def parameter_set(self, row, loss):
        if self.option==0: return loss/np.dot(row, row)
        if self.option==1: return np.minimum(self.C, loss/np.dot(row, row))
        if self.option==2: return loss/((1/(2*self.C)) + np.dot(row, row))
        
        
    def sparsity_step(self):
        projected= np.multiply(np.minimum(1, self.Lambda/np.linalg.norm(self.weights, ord=1)), self.weights)
        self.weights= self.truncate(projected)
        
        
    def truncate(self, projected):
        if np.linalg.norm(projected, ord=0)> self.B*len(projected):
            remaining= int(np.maximum(1, np.floor(self.B*len(projected))))
            for i in projected.argsort()[:(len(projected)-remaining)]:
                projected[i]=0
            return projected
        else: return projected


    def fit(self, row, y_hat, y):
        if self.wts_init_flag == False:
            self.set_classifier(row)
            self.wts_init_flag = True
            
        if y_hat != y:
            loss = (np.maximum(0, 1 - y*y_hat))
            tao = self.parameter_set(row, loss)
            w_1 = self.weights + np.multiply(tao * y, row[:len(self.weights)])
            w_2= np.multiply(tao * y, row[len(self.weights):])
            self.weights= np.append(w_1, w_2)
            #self.sparsity_step()
            
            
    def predict(self, row):
        if self.wts_init_flag == True:
            return np.sign(np.dot(self.weights, row[:len(self.weights)]))
        else:
            return -1 # neither -1 nor 1, for fairness.




