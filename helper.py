
import json
import matplotlib.pyplot as plt
import torch
import os
import numpy as np

# save the training parameters in a txt at the beginning of training	
def save_params(par, model_dir, name):
    data = {}
    for att in dir(par):
        if not att.startswith('__'):
            data[att] = par.__getattribute__(att)
    with open(model_dir + "parameters_"+name+".json", 'w') as outfile:  
        json.dump(data, outfile, indent=4)


def plot_loss(loss, epoch, iteration, step, loss_name, loss_dir, avg=False):
    #x = range(1,iteration, step)
    x = list(range(iteration))
    #y = loss[1:iteration:step]
    y = loss
    #print "loss_name:", loss_name, " y: ", y
    #print "loss ", loss[1:iteration:step]
    plt.plot(x, y, 'b')
    if avg: # plot the average of every k elements
        k=10
        if iteration>=k:
            loss = np.asarray(loss, dtype=np.float32)
            #print(loss.shape)
            loss_mean = np.mean(loss.reshape(-1,k), axis=1)
            #print(loss_mean.shape)
            x_mean = list(range(k-1,iteration, k))
            plt.plot(x_mean, loss_mean, 'r')

    #plt.axis([0, iteration, 0, max(loss[1:iteration:step])+0.1])
    plt.axis([0, iteration, 0, max(loss)])
    plt.ylabel(loss_name + ' Loss')
    plt.xlabel('Iter')
    plt.title(loss_name)
    #plt.show()
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)	
    plt.savefig(loss_dir+loss_name+"_"+str(epoch)+"_"+str(iteration)+"_loss.png")
    plt.clf()



def save_model(model, model_dir, model_name, train_iter):
    model_path = model_dir+model_name+"_"+str(train_iter)+".pt"
    torch.save(model, model_path)
    print("Saved:", model_path)

def load_model(model_dir, model_name, test_iter, eval=True):
    model_path = model_dir+model_name+"_"+str(test_iter)+".pt"
    model = torch.load(model_path)
    print("Loaded model:", model_path)
    model.cuda()
    if eval:
        model.eval()
    else:
        model.train()
    return model