import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    m,n =X.shape
    w=np.zeros(n)
    b=0.0
    
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    
        
    for i in range(steps):
        
        y1=X@w+b
        p=_sigmoid(y1)
        loss=p-y
        gradient_w=1/m*(X.T@loss)
        gradient_b=1/m*np.sum(loss)
        w=w-lr*gradient_w
        b=b-lr*gradient_b
            
        
    
    return w,b
    pass