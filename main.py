import torch.optim as optim

from torch.autograd import Variable
import torch.nn as nn

from data_handler.loaders import get_data_loaders
from model import LeNet


def shit():
    # defin hyperparameters
    lr = 0.001
    epochs = 8

    output_size = 30
    model = LeNet(output_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):

        # move on the train
        model.train()
        for batch_idx, (input, label) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(Variable(input))
            loss = nn.functional.nll_loss(output, Variable(label))
            loss.backward()
            optimizer.step()

        # move on the validation
        model.eval()

        correct = 0
        for data, target in valid_loader:
            output = model(Variable(data, volatile=True))
            target = Variable(target)
            y_hat = output.data.max(1, keepdim=True)[1]

    # move on the test
    fout = open('test_y', 'w')
    model.eval()
    file_ind = 0
    files = test_dataset.files
    for data, target in test_loader:
        output = model(Variable(data, volatile=True))
        y_hat = output.data.max(1, keepdim=True)[1]
        list_y_hat = y_hat.tolist()
        for i in range(len(list_y_hat)):
            fout.write(str(files[file_ind]) + ", " + str(list_y_hat[i][0]) + '\n')
            file_ind += 1

    fout.close()


def main():
    train_loader, valid_loader, test_loader = get_data_loaders()
    for epoch in range(epochs):
        train_loss = train(train_loader)
        val_loss, val_acc = evaluation(valid_loader)
    test_acc = test(test_loader)
    plot_graphs(train_loss, val_loss, val_acc, test_acc)



if __name__ == "__main__":
    main()
