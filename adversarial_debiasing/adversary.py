import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import torch.optim as optim

from adversarial_dataset import get_adversary_dataloader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HIDDEN_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001


class baseline_classifier(nn.Module):
    def __init__(self, num_classes = 8):
        super(baseline_classifier, self).__init__()
        self.model_ft= resnet50(pretrained = ResNet50_Weights)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, num_classes)

        self._model_ft = self._model_ft.to(DEVICE)

    def forward(self, x):
        # forward through linear layers
        out = self.model_ft(x)
        return out

class adversary_classifier(nn.Module):
    def __init__(self, input_size = 8, hidden_size = HIDDEN_SIZE, num_classes = 5):
        super(adversary_classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x


def classifier_train(classifier, adversary,
                    train_loader, val_loader,
                    optimizer_cls, optimizer_adv,
                    num_epochs, device = DEVICE):
    for epoch in range(num_epochs):
        classifier.train()
        adversary.train()
        for batch_idx, (data, label, group) in enumerate(train_loader):
            data, label, group = data.to(device), label.to(device), group.to(device)
            optimizer_cls.zero_grad()
            preds = classifier(data)
            loss_cls = F.cross_entropy(preds, label)
            loss_cls.backward()
            optimizer_cls.step()

            optimizer_adv.zero_grad()
            output_adv = adversary(label)
            loss_adv = F.cross_entropy(output_adv, group)
            loss_adv.backward()
            optimizer_adv.step()

        classifier.eval()
        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for (data, label, group) in val_loader:
                data, label, group = data.to(device), label.to(device), group.to(device)
                label_preds = classifier(data)
                output_adv = adversary(label_preds)
                pred_label = torch.argmax(output_adv, dim=1)
                val_loss += F.cross_entropy(pred_label, label)
                val_acc += (pred_label == label).float().mean()
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
        print(f"Epoch {epoch+1}: Training Loss={loss_cls.item():.4f}, \
            Validation Loss={val_loss.item():.4f}, Validation Accuracy={val_acc.item()*100:.2f}%")

def test(classifier, adversary, test_loader, device = DEVICE):
    classifier.eval()

    test_loss = 0
    test_acc = 0

    with torch.no_grad():
        for (img, label, group) in test_loader:
            data, label, group = data.to(device), label.to(device), group.to(device)
            output = classifier(img)
            adv_output = adversary(output)
            adv_pred = torch.argmax(adv_output, dim=1)
            debiased_pred = torch.zeros_like(adv_pred)

            for i in range(len(debiased_pred)):
                debiased_pred[i] = output[i, adv_pred[i]]

            debiased_probs = torch.nn.functional.softmax(debiased_pred, dim=1)
            debiased_labels = torch.argmax(debiased_probs, dim=1)

            test_loss += F.cross_entropy(debiased_labels, label)
            test_acc += (debiased_labels == label).float().mean()

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)

    print(f"Test Loss={test_loss.item():.4f}, Test Accuracy={test_acc.item()*100:.2f}%")

def main(args):

    # Define the loss function for the adversarial classifier
    baseline = baseline_classifier()
    adversary = adversary_classifier()

    # Define the optimizer and hyparameters for training the adversarial classifier
    optimizer_base = optim.Adam(baseline.parameters(), lr = LEARNING_RATE)
    optimizer_adv = optim.Adam(adversary.parameters(), lr = LEARNING_RATE)

    # Define dataloaders
    train_loader = get_adversary_dataloader(data_csv = args.train_csv,
                split = 'train',
                batch_size = args.batch_size,
                task_batch_size = None)

    val_loader = get_adversary_dataloader(data_csv = args.val_csv,
                split = 'val',
                batch_size = args.batch_size,
                task_batch_size = None)

    classifier_train(baseline, adversary, train_loader, val_loader,
                     optimizer_base, optimizer_adv, num_epochs = NUM_EPOCHS)
    test(baseline, adversary, test_loader = get_adversary_dataloader(data_csv = args.test_csv,
                split = 'test',
                batch_size = args.batch_size,
                task_batch_size = None))