import torch
import torch.optim as optim
from gcommand_loader import GCommandLoader
from torch.autograd import Variable
import torch.nn as nn

kernel_conv = 5
kernel_mp = 2

conv_out1 = 20
conv_out2 = 20
h = int((((161 - kernel_conv + 1) / 2) - kernel_conv + 1) / 2)
w = int((((101 - kernel_conv + 1) / 2) - kernel_conv + 1) / 2)
fc1_in = h * w * conv_out2


class LeNet(nn.Module):
    def __init__(self, output_size):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, conv_out1, kernel_size=kernel_conv)
        self.max_pool1 = nn.MaxPool2d(kernel_size=kernel_mp)
        self.conv2 = nn.Conv2d(conv_out1, conv_out2, kernel_size=kernel_conv)
        self.max_pool2 = nn.MaxPool2d(kernel_size=kernel_mp)
        self.fc1 = nn.Linear(fc1_in, 1000)
        self.fc2 = nn.Linear(1000, output_size)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = nn.functional.log_softmax(x)
        return x


# load the data
train_dataset = GCommandLoader('data/train')
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=100, shuffle=True,
    num_workers=20, pin_memory=True, sampler=None)

valid_dataset = GCommandLoader('data/valid')
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=100, shuffle=None,
    num_workers=20, pin_memory=True, sampler=None)

test_dataset = GCommandLoader('data/test', is_test=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=100, shuffle=None,
    num_workers=20, pin_memory=True, sampler=None)

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
