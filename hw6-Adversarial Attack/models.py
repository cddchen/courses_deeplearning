import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F


class Attacker:
    def __init__(self, x, y):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.vgg16(pretrained=True)
        self.model.to(device)
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        self.dataset = CustomImageDataSet(x, y, preprocess)
        self.loader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False
        )

    def fgsm_attack(self, img, epsilon, data_grad):
        # find the direction of gradient
        sign_data_grad = data_grad.sign()
        perturbed_img = img + epsilon * sign_data_grad
        return perturbed_img

    def attack(self, epsilon):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        adv_examples = []
        wrong, fail, success = 0, 0, 0
        for data, target in self.loader:
            data, target = data.to(device), target.to(device)
            data_raw = data
            data.requires_grad = True
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]

            if init_pred.item() != target.item():
                wrong += 1
                continue

            loss = F.nll_loss(output, target)
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

            output = self.model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]

            if final_pred.item() == target.item():
                fail += 1
            else:
                success += 1
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data * torch.tensor(self.std, device=device).view(3, 1, 1) + torch.tensor(
                        self.mean, device = device).view(3, 1, 1)
                    adv_ex = adv_ex.squeeze().detach().cpu().numpy()
                    data_raw = data_raw * torch.tensor(self.std, device=device).view(3, 1, 1) + torch.tensor(
                        self.mean, device=device).view(3, 1, 1)
                    data_raw = data_raw.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), data_raw, adv_ex))
        final_acc = (fail / (wrong + success + fail))

        print('Epsilon: %f\tTest Accuracy: %d / %d = %.2f' % (epsilon, fail, len(self.loader), final_acc))
        return adv_examples, final_acc