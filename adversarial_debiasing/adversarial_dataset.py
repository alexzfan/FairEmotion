import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, ColorJitter, RandomHorizontalFlip
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight

TRANSFORM = Compose([
                ToTensor(),
                Resize((224,224)),
                Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ]
        )

class AdversarialDataset(Dataset):
    def __init__(self, data_file, train, transform = TRANSFORM):
        super().__init__()
        # read data file
        self._data = pd.read_csv(data_file)
        self._race_idx = dict()
        for i, race in enumerate(np.unique(self._data.race)):
            self._race_idx[race] = i

        self.label_weights= compute_class_weight(class_weight='balanced', classes= np.unique(self._data.label), y= np.array(self._data.label))
        
        self.train = train
        self.transform = transform

    def __getitem__(self, index):
        
        image_path = self._data.loc[index, 'img_path']
        image = Image.open(image_path).convert("RGB")

        if self.train:
            std_image = transforms.Compose(
                [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                )
                ]
            )
        else:
            std_image = transforms.Compose(
                [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                )
                ]
            )
        image = std_image(image)

        assert(image.shape == (3,244,244))

        label = self._data.loc[index, "label"]
        image_num = self._data.loc[index, 'image_num']
        race = self._race_idx[self._data.loc[index,'race']]

        example = (
            image_num,
            image,
            label,
            race
        )

        return example

def get_adversary_dataloader(data_csv, split, batch_size):
    if split == "train":
        dataset = AdversarialDataset(data_csv, train = True)
        return DataLoader(dataset, 
                            batch_size = batch_size,
                            shuffle = True,
                            num_workers = 8)

    else:
        dataset = AdversarialDataset(data_csv, train = False)
        return DataLoader(dataset, 
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = 8)
def acc_score(logits, labels):
    """Returns the mean accuracy of a model's predictions on a set of examples.
    Args:
        logits (torch.Tensor): model predicted logits
            shape (examples, classes)
        labels (torch.Tensor): classification labels from 0 to num_classes - 1
            shape (examples,)
    """

    assert logits.dim() == 2
    assert labels.dim() == 1
    assert logits.shape[0] == labels.shape[0]

    preds = torch.argmax(logits, dim = -1)
    corrects_bool = preds == labels
    corrects_bool = corrects_bool.type(torch.float)

    num_correct = torch.sum(corrects_bool).item()
    acc = torch.mean(corrects_bool).item()
    return preds, num_correct, 

def evaluate(model, data_loader, device):
    nll_meter = AverageMeter()

    model.eval()
    pred_dict = {} # id, prob and prediction
    full_labels = []
    predictions = []
    race_labs = data_loader.dataset.data['race']
    test = []

    acc = 0
    num_corrects, num_samples = 0, 0

    with torch.no_grad():
        for img_id, x, y, race in data_loader:
            # forward pass here
            x = x.float().to(device)

            batch_size = x.shape[0]

            score = model(x)

            # calc loss
            y = y.type(torch.LongTensor).to(device)
            criterion = nn.CrossEntropyLoss()

            preds, num_correct, acc = acc_score(score, y) 
            loss = criterion(score, y)

            loss_val = loss.item() 
            nll_meter.update(loss_val, batch_size)

            # get acc and auroc
            num_corrects += num_correct
            num_samples += preds.size(0)
            predictions.extend(preds)
            full_labels.extend(y)
            test.extend(race)


        acc = float(num_corrects) / num_samples

        # F1 Score
        y_pred = np.asarray([pred.cpu() for pred in predictions]).astype(int)
        y = np.asarray([label.cpu() for label in full_labels]).astype(int)
        f1 = metrics.f1_score(y, y_pred, average = 'weighted')

        # dem parity ratio
        fairlearn_dem_parity_ratio = demographic_parity_ratio(y_true = y, 
                                                    y_pred = y_pred, 
                                                    sensitive_features = race_labs)

        # weighted OVO fairness metrics
        truth_sample_size_weights = pd.Series(y).value_counts(normalize = True).sort_index().tolist()
        assert(len(truth_sample_size_weights) == 7)

        dem_parity_ratio = 0
        eq_odds_ratio = 0

        for i, weight in enumerate(truth_sample_size_weights):
            y_true_converted = [1 if j == i else 0 for j in y]
            y_pred_converted = [1 if j == i else 0 for j in y_pred]

            dem_parity_ratio += weight*demographic_parity_ratio(y_true = y_true_converted, 
                                            y_pred = y_pred_converted,
                                            sensitive_features = race_labs
                                            )
            eq_odds_ratio += weight*equalized_odds_ratio(y_true = y_true_converted,
                                                        y_pred = y_pred_converted,
                                                        sensitive_features = race_labs,
                                                        )
        
    model.train()

    results_list = [("NLL", nll_meter.avg),
                    ("Acc", acc),
                    ("F1 Score", f1),
                    ("FL Dem Parity Ratio", fairlearn_dem_parity_ratio),
                    ("OVO Dem Parity Ratio", dem_parity_ratio),
                    ("OVO Eq Odds Ratio", eq_odds_ratio),]
    results = OrderedDict(results_list)
    return results, pred_dict