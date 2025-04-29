# 🧠📸 OptiPredict: Smart Fundus Imaging and Automated Ocular Disease Diagnosis

OptiPredict is a dual-solution system combining a **portable, nonmydriatic fundus camera prototype** with a **deep learning-powered diagnostic platform**. Designed for affordability, mobility, and diagnostic accuracy, this system empowers ophthalmologists and healthcare providers in **resource-constrained** environments.

---

## 🔬 Project Highlights

### 📷 1. Portable Fundus Camera Prototype
A **low-cost, handheld**, nonmydriatic fundus camera engineered using:

- **Raspberry Pi** with IR-sensitive camera module
- Dual **infrared and white-light LEDs**
- **5-inch touchscreen LCD**
- **Rechargeable battery** for portability
- **Disposable 20D condensing lens**

> 💡 **Specs**: 133mm × 91mm × 45mm, **386g**, **₹15,374**

#### ✅ Key Benefits:
- Captures **high-quality retinal images** without pharmacologic dilation
- Compact, lightweight, and ideal for **rural or mobile screenings**
- Usable by healthcare workers with minimal training

---

### 🤖 2. AI-Powered Ocular Disease Detection (OptiPredict)

A deep learning-based diagnostic platform trained on **fundus images** to detect eye diseases **automatically and in real time**.

#### 🧠 Models Implemented:
- **ResNet-50** – 98% validation accuracy
- **VGG-19** – 94.44% validation accuracy
- **Vision Transformer** – 85.71% validation accuracy

#### 🧰 Core Technologies:
- **Local Binary Patterns (LBP)** for enhanced feature extraction
- **Binary classification** to mitigate class imbalance
- **ODIR dataset** (6,000 labeled fundus images)
- **Cloud-based architecture** for deployment & real-time diagnostics

#### 🔁 Preprocessing Pipeline:
- Image normalization, resizing, and augmentation
- Feature enhancement using LBP
- Split into training/validation sets for robust evaluation

