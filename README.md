this project utiizes a neural net to analyze and classify the FashionMNIST dataset
uses softmax function to predict labels on the test set. 
Input test tensors are N*1*1*28 images
output is the most likely 3 class labels

The layers of the model are as follows:
1) Flatten layer to conver 2D pixel to 1D array
2) Dense layer with 128 nodes and a reLu activation
3) Dense layer with 64 nodes and a reLu activation
4) Dense layer with 10 nodes
   a sequential container is used to hold these labels. 
