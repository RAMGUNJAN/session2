
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # TODO: Defining model architecture here
        self.fc1 = torch.nn.Linear(784, 256)
        self.fc2 = torch.nn.Linear(256, 10)

    def forward(self, x):
        # TODO: Defining the forward pass
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_epoch(epoch, args, model, device, data_loader, optimizer):
    # Set the model to training mode
    model.train()

    for batch_idx, (data, target) in enumerate(data_loader):
        # Move data and target tensors to the device (GPU or CPU)
        data, target = data.to(device), target.to(device)

        # Reset the gradients
        optimizer.zero_grad()

        # Perform the forward pass
        output = model(data)

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Perform backpropagation
        loss.backward()

        # Update the weights
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} '
                  f'({100. * batch_idx / len(data_loader):.2f}%)]\tLoss: {loss.item():.6f}')

def test_epoch(model, device, data_loader):
    # Set the model to evaluation mode
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            # Move data and target tensors to the device (GPU or CPU)
            data, target = data.to(device), target.to(device)

            # Perform the forward pass
            output = model(data)

            # Calculate the loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            # Get the predicted labels
            pred = output.argmax(dim=1, keepdim=True)

            # Count the correctly classified samples
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} '
          f'({accuracy:.2f}%)\n')

def main():
    # Parser to get command line arguments
    parser = argparse.ArgumentParser(description='MNIST Training Script')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Define the data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the MNIST dataset for training and testing
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=1)

    model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Training and testing cycles
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, device, train_loader, optimizer)
        test_epoch(model, device, test_loader)

        # Save the model checkpoint after each epoch
        torch.save(model.state_dict(), f'model_checkpoint_epoch_{epoch}.pth')

if __name__ == "__main__":
    main()




