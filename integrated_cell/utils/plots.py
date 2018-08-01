
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

import pdb

dpi = 100
figx = 6
figy = 4.5

def history(logger, save_path):
    
    #Figure out the default color order, and use these for the plots
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.figure(figsize=(figx, figy), dpi=dpi)
    
    ax = plt.gca()
    
    plts = list()
    
    #Plot reconstruction loss
    plts += ax.plot(logger.log['iter'], logger.log['reconLoss'], label='reconLoss', color=colors[0])
    
    plt.ylabel('reconLoss')
    
    ax_max = np.percentile(logger.log['reconLoss'], 99.5)
    ax_min = np.percentile(logger.log['reconLoss'], 0)
                          
    ax.set_ylim([ax_min,ax_max])
    
    #Plot everything else that isn't below
    do_not_print= ['epoch', 'iter', 'time', 'reconLoss']
    
    #Print off the reconLoss on it's own scale
    ax2 = plt.gca().twinx()
    i = 1
    for field in logger.fields:
        if field not in do_not_print:
            plts += ax2.plot(logger.log['iter'], logger.log[field], label=field, color=colors[i])
            i += 1

    #Get all the labels for the legend from both axes
    labs = [l.get_label() for l in plts]

    #Print legend
    ax.legend(plts, labs)
    
    plt.ylabel('loss')
    plt.title('History')
    plt.xlabel('iteration')    
    
    #Save
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close()

def short_history(logger, save_path, max_history_len = 10000):
    history = int(len(logger.log['epoch'])/2)

    if history > max_history_len:
        history = max_history_len

    x = logger.log['iter'][-history:]
    y = logger.log['reconLoss'][-history:]

    epochs = np.floor(np.array(logger.log['epoch'][-history:]))
    losses = np.array(logger.log['reconLoss'][-history:])
    iters = np.array(logger.log['iter'][-history:])
    uepochs = np.unique(epochs)

    epoch_losses = np.zeros(len(uepochs))
    epoch_iters = np.zeros(len(uepochs))
    i = 0
    for uepoch in uepochs:
        inds = np.equal(epochs, uepoch)
        loss = np.mean(losses[inds])
        epoch_losses[i] = loss
        epoch_iters[i] = np.mean(iters[inds])
        i+=1

    mval = np.mean(losses)

    plt.figure(figsize=(figx, figy), dpi=dpi)
    plt.plot(x, y, label='reconLoss')
    plt.plot(epoch_iters, epoch_losses, color='darkorange', label='epoch avg')
    plt.plot([np.min(iters), np.max(iters)], [mval, mval], color='darkorange', linestyle=':', label='window avg')

    plt.legend()
    plt.title('Short history')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close()
    
def embeddings(embedding, save_path):
    plt.figure(figsize=(figx, figy), dpi=dpi)
    colors = plt.get_cmap('plasma')(np.linspace(0, 1, embedding.shape[0]))
    plt.scatter(embedding[:,0], embedding[:,1], s = 2, color = colors)
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.axis('equal')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.title('latent space embedding')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close()

