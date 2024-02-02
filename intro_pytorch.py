import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    """
    #if training=true, return dataloader to training set, else return the test set
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    tset=datasets.FashionMNIST('./data',
            train=training,
            download=training,
            transform=custom_transform)
    tloader=torch.utils.data.DataLoader(tset,
            batch_size=64,
            shuffle=False)
    return tloader


def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    # flatten layer to convert the 2D pixel array to a 1D array
    # dense layer with 128 nodes and a ReLU activation
    # dense layer with 64 nodes and a ReLU activation
    # dense layer with 10 nodes
    # sequential container to hold these layers
    model=nn.Sequential(nn.Flatten(),
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10))
    return model






def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
    model.train()
    epoch=T
    for e in range(epoch):
        running_loss=0.0
        correct_preds=0
        for i,data in enumerate(train_loader,0):
            input,labels=data
            opt.zero_grad()
            outputs=model(input)
            _, predicted=torch.max(outputs.data,1)
            correct_preds+=(predicted==labels).sum().item()
            loss=criterion(outputs,labels)
            loss.backward()
            opt.step()
            running_loss+=input.size(0)*loss.item()
        avg_loss=running_loss/len(train_loader.dataset)
        accuracy=100. * correct_preds/len(train_loader.dataset)
        print(f'Train Epoch: {e} Accuracy: {correct_preds}/{len(train_loader.dataset)}({accuracy:.2f}%) Loss: {avg_loss:.3f}')



def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    with torch.no_grad():
        running_loss=0.0
        correct_preds=0
        for i,data in enumerate(test_loader,0):
            input,labels=data
            outputs=model(input)
            _, predicted=torch.max(outputs.data,1)
            correct_preds+=(predicted==labels).sum().item()
            loss=criterion(outputs,labels)
            #loss.backward()
            running_loss+=input.size(0)*loss.item()
        avg_loss=running_loss/len(test_loader.dataset)
        accuracy=100. * correct_preds/len(test_loader.dataset)
        if show_loss:
            print(f'Average Loss: {avg_loss:.4f}')
        print(f'Accuracy: {accuracy:.2f}%')

def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                   'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    # Get the image at the specified index
    image = test_images[index].unsqueeze(0).float() # Add a batch dimension

    # Ensure we are in evaluation mode
    model.eval()

    # Get the logits from the model
    logits = model(image)

    # Convert logits to probabilities
    probs = F.softmax(logits, dim=1)

    # Get the top 3 probabilities and their indices
    top_probs, top_indices = probs.topk(3)

    # Convert the probabilities and indices to lists
    top_probs = top_probs.detach().numpy()[0]
    top_indices = top_indices.detach().numpy()[0]
    for i in range(3):
        print(f"{class_names[top_indices[i]]}: {top_probs[i]*100:.2f}%")


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    train_loader=get_data_loader()
    test_loader=get_data_loader(False)
    model=build_model()
    print(model)
    criterion = nn.CrossEntropyLoss()
    train_model(criterion=criterion, model=model, train_loader=train_loader,T=5)
    evaluate_model(model,test_loader,criterion,show_loss=False)
    evaluate_model(model,test_loader,criterion)
    predict_label(model,test_loader.dataset.data,1)
