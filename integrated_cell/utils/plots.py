
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

def history(logger, save_path):
    plt.figure()
    
    for i in range(2, len(logger.fields)-1):
        field = logger.fields[i]
        plt.plot(logger.log['iter'], logger.log[field], label=field)

    plt.legend()
    plt.title('History')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig(save_path, bbox_inches='tight')
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

    plt.figure()
    plt.plot(x, y, label='reconLoss')
    plt.plot(epoch_iters, epoch_losses, color='darkorange', label='epoch avg')
    plt.plot([np.min(iters), np.max(iters)], [mval, mval], color='darkorange', linestyle=':', label='window avg')

    plt.legend()
    plt.title('Short history')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
def embeddings(embedding, save_path):
    plt.figure()
    colors = plt.get_cmap('plasma')(np.linspace(0, 1, embedding.shape[0]))
    plt.scatter(embedding[:,0], embedding[:,1], s = 2, color = colors)
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.axis('equal')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.title('latent space embedding')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

