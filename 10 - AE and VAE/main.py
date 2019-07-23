import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets



def  main():

    mnist_train = datasets.MNIST('mnist', train=True, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_train = DataLoader(mnist, batch_size=32, shuffle=True)

    mnist_test = datasets.MNIST('mnist', train=False, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_test = DataLoader(mnist, batch_size=32, shuffle=True)

    x, _ = iter(mnist_train).next()



if __name__ == "__main__":
    main()