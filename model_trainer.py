import torch

class Trainer:
    def __init__(self, train_loader, validation_loader, model, criterion, optimizer, device, writer):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.writer = writer
    
    def train(self, epochs):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                print(loss)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if i % 200 == 199:
                    cls_acc = self.classification_accuracy('train')
                    av_loss = running_loss / 200
                    self.writer.add_scalar("Loss/train", av_loss, epoch)
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {av_loss:.3f} training classification_accuracy: {cls_acc:.3f}')
                    running_loss = 0.0
                    self.validation(epoch)

        print('Finished Training')


    def classification_accuracy(self, dataloader_type):
        if dataloader_type == 'train':
            dataloader = self.train_loader
        elif dataloader_type == 'validation':
            dataloader = self.validation_loader
        else:
            raise ValueError('dataloader_type must be either train or validation')

        correct = 0
        total = 0
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total


    def validation(self, train_epoch):
        total = 0
        correct = 0
        running_loss = 0.0
        for i, data in enumerate(self.validation_loader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        val_acc = correct / total
        av_loss = running_loss / len(self.validation_loader)
        self.writer.add_scalar("loss/validation", av_loss, train_epoch)
        self.writer.add_scalar("accuracy/validation", val_acc, train_epoch)

        print(f'[{train_epoch + 1}, {i + 1:5d}] validation loss: {av_loss:.3f} validation classification_accuracy: {val_acc:.3f}')
