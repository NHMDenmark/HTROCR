import torch
import torch.nn as nn
from tqdm import tqdm
import utils.device as device
from data_processors.emnist_data_processor import prepare_data
from models.dcnn_ctc import DCNNEncoder
import matplotlib.pyplot as plt

@torch.no_grad()
def evaluate(model, val_loader, criterion):
    model.eval()
    outputs = [model.validation_step(batch, criterion) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, max_lr, model, train_loader, val_loader, 
          weight_decay=0, grad_clip=None, opt_func=torch.optim.Adam):
    torch.cuda.empty_cache()
    history = []
    
    criterion = nn.CTCLoss(blank=10, zero_infinity=True)
    # Set up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader, position=0, leave=True):
            loss = model.training_step(batch, criterion)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        
        # Validation phase
        result = evaluate(model, val_loader, criterion)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def train():
    epochs = 5
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam
    model = device.to_device(DCNNEncoder(), device.default_device())
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")
    train_dl, valid_dl = prepare_data()
    history = []
    history += fit(epochs, max_lr, model, train_dl, valid_dl, 
                grad_clip=grad_clip, 
                weight_decay=weight_decay, 
                opt_func=opt_func)
    return history

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.savefig("dcnn_accuracy.png")

if __name__ == '__main__':
    history = train()
    plot_accuracies(history)