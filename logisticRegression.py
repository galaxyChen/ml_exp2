# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    s = 1/(1+np.exp(-z.A1))
    s = np.array(list(map(lambda x : x,s)))
    return s

def loss(X,y,w,lamb):
    h = sigmoid(X*w)
    m = y.shape[0]
    y_temp = np.array(list(map(lambda x:1 if x[0]==1 else 0,y.A)))
    J = -y_temp*np.log(h)-(1-y_temp)*np.log(1-h)
    w_reg = w
    w_reg[0]=0
    return J.sum()/m+lamb/(2*m)*((w_reg.A**2).sum())

def gradient(X,y,w,lamb):
    m = y.shape[0]
    y_temp = np.array(list(map(lambda x:1 if x[0]==1 else 0,y.A)))
    h_y = np.matrix(sigmoid(X*w)-y_temp)
    dJ = X.T*(h_y.T)/m
    w_reg = w
    w_reg[0] = 0
    g = dJ+w_reg*lamb/m
    return g
    
def gradientDecent(X,y,w,alpha,lamb,num_rounds,val_x,val_y):
    train_loss_history = []
    val_loss_history = []
    print("origin train loss:%f"%loss(X,y,w,lamb))
    train_loss_history.append(loss(X,y,w,lamb))
    print("origin validation loss:%f"%loss(val_x,val_y,w,lamb))
    val_loss_history.append(loss(val_x,val_y,w,lamb))
    print("")
    
    for i in range(num_rounds):
        w = w - gradient(X,y,w,lamb)*alpha
        train_loss_history.append(loss(X,y,w,lamb))
        val_loss_history.append(loss(val_x,val_y,w,lamb))
        
    return w,train_loss_history,val_loss_history

def train(X,y,val_x,val_y):
    m = X.shape[1]
    init_w = np.matrix(np.zeros(m)).T
    print("begin to train")
    alpha=0.1
    num_rounds=500
    lamb = 1
    print("learning rate alpha:%f"%alpha)
    print("number of rounds:%d"%num_rounds)
    print("lambda : %f"%lamb)
    print("")
    w,train_loss_history,val_loss_history = gradientDecent(X,y,init_w,alpha,lamb,num_rounds,val_x,val_y)
    plt.plot(np.arange(num_rounds+1),train_loss_history,label='train loss')
    plt.plot(np.arange(num_rounds+1),val_loss_history,label='validation loss')
    plt.legend(loc=1)
    plt.xlabel('number_of_rounds')
    plt.ylabel('loss')
    return w,train_loss_history,val_loss_history
    
def predict(X,y,w):
    pred = sigmoid(X*w)
    num_p = (y==1).sum()
    num_n = (y==-1).sum()
    pred = pred/(1-pred)
    pred_y = list(map(lambda x:1 if x>num_p/num_n else -1,pred))
    acc = (y.A1==pred_y).sum()/len(y.A)
    print("acc:%f"%acc)


def getData():
    X,y = datasets.load_svmlight_file('./a9a',n_features=123)
    X = np.matrix(X.toarray())
    ones = np.matrix(np.ones((X.shape[0],1)))
    train_x = np.concatenate((ones,X),axis=1)
    train_y = np.matrix(y).T
    
    X,y = datasets.load_svmlight_file('./a9a.t',n_features=123)
    X = np.matrix(X.toarray())
    ones = np.matrix(np.ones((X.shape[0],1)))
    test_x = np.concatenate((ones,X),axis=1)
    test_y = np.matrix(y).T
    return train_x,test_x,train_y,test_y
    

train_x,test_x,train_y,test_y = getData()
w,train_loss,val_loss = train(train_x,train_y,test_x,test_y)
print("final train loss:%f"%train_loss.pop())
print("final validation loss:%f"%val_loss.pop())
predict(train_x,train_y,w)
predict(test_x,test_y,w)


