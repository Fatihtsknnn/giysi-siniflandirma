# 👕 Giysi Sınıflandırma — CNN (FashionMNIST)

PyTorch ile sıfırdan yazılmış bir Evrişimli Sinir Ağı (CNN). 
28x28 gri tonlamalı giysi görüntülerini 10 kategoriye ayırır.

## 📊 Sonuç
- **Test Doğruluğu: %85.3**
- Veri seti: FashionMNIST (60.000 eğitim + 10.000 test görüntüsü)
- Eğitim: 1 epoch, Adam optimizer (lr=0.001)

## 🧠 Model Mimarisi



## 🏷️ Sınıflar
Tişört, Pantolon, Kazak, Elbise, Mont, Sandalet, Gömlek, Spor Ayakkabı, Çanta, Bot

## 🚀 Çalıştırma
```bash
pip install torch torchvision matplotlib
python giysi.py
```

## 📁 Dosyalar
- `giysi.py` — Veri yükleme, model, eğitim ve değerlendirme
- `tahminler.png` — 8 test görüntüsü üzerinde tahmin/gerçek karşılaştırması

## 🛠️ Kullanılan Teknolojiler
PyTorch · torchvision · Matplotlib