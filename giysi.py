import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 

transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(root="veri",train=True,download=True,transform=transform)
test_data = datasets.FashionMNIST(root="veri",train=False,download=True,transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data,batch_size=64)

print("Egitim örnek sayisi", len(train_data))
print("Test örnek sayisi", len(test_data))

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32*7*7,10)

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))  
        x = x.view(x.size(0),-1)
        x = self.fc(x)  
        return x 

model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

print("Egitim basliyor...")
model.train()
for epoch in range(1):
    for resimler, etiketler in train_loader:
        tahminler = model(resimler)
        loss = loss_fn(tahminler,etiketler)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} bitti, Son Loss: {loss.item():.4f}")

model.eval()
dogru = 0
toplam = 0
with torch.no_grad():
    for resimler, etiketler in test_loader:
        tahminler = model(resimler)
        _, tahmin = torch.max(tahminler,1)
        toplam += etiketler.size(0)
        dogru += (tahmin == etiketler).sum().item() 

print(f"\nTest Dogrulugu: %{100 * dogru / toplam: .2f}")

sinif_isimleri = ["Tişört","Pantalon","Kazak","Elbise","Mont","Sandalet","Gömlek","Spor Ayakkabi","Çanta","Bot"]

ornekler, gercek = next(iter(test_loader))
with torch.no_grad():
    ciktilar = model(ornekler)
    _, tahminler = torch.max(ciktilar, 1)

plt.figure(figsize=(14,5))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(ornekler[i][0],cmap="gray")
    t = sinif_isimleri[tahminler[i].item()]
    g = sinif_isimleri[gercek[i].item()]    
    plt.title(f"tahmin: {t}\nGerçek: {g}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("tahminler.png")    
plt.show()
print("Görselleştirme kaydedildi!")

