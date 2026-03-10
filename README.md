<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0D1117,40:1a237e,100:7c4dff&height=160&text=CIFAR-10%20Image%20Classifier&fontSize=28&fontColor=ffffff&animation=fadeIn&fontAlignY=65" width="100%"/>

<h1 align="center">🧠 CIFAR-10 Image Classifier</h1>
<h3 align="center">Deep Neural Network with Batch Normalization & Dropout</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white" alt="Matplotlib"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge" alt="Status"/>
  <img src="https://img.shields.io/badge/Author-MH--SHUVO20-7c4dff?style=for-the-badge&logo=github" alt="Author"/>
</p>

> **Designed, developed, and owned by [MD. MEHEDI HASAN SHUVO](https://github.com/MH-SHUVO20)**

A complete end-to-end deep learning project that builds, trains, and evaluates a **Dense Neural Network (MLP)** on the CIFAR-10 image classification benchmark. The project explores what baseline accuracy is achievable with purely fully-connected layers — without any Convolutional Neural Network (CNN) — serving as a foundational study in deep learning concepts including Batch Normalization, Dropout regularization, and the Adam optimizer.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Libraries Used](#-libraries-used)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Training Configuration](#-training-configuration)
- [Results & Performance](#-results--performance)
- [Key Findings](#-key-findings)
- [Getting Started](#-getting-started)
- [License](#-license)
- [Author](#-author)

---

## 🎯 Project Overview

This project performs a full supervised deep learning pipeline on the CIFAR-10 dataset to classify 32×32 color images into one of 10 categories using a **Multilayer Perceptron (MLP)** — a fully connected Dense Neural Network.

**Objective:** Demonstrate the baseline performance of a Dense Neural Network on image classification, understand its limitations on spatially-structured data, and highlight *why* Convolutional architectures are needed for high-accuracy image classification.

**Key Questions Explored:**

- What accuracy ceiling does a fully-connected MLP achieve on CIFAR-10?
- How do Batch Normalization and Dropout contribute to training stability and regularization?
- How do training loss and validation loss curves reveal overfitting behavior?
- What is the fundamental difference between a dense network and a CNN for image data?

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Dataset Name** | CIFAR-10 |
| **Source** | `tensorflow.keras.datasets.cifar10` (auto-downloaded) |
| **Total Images** | 60,000 color images |
| **Training Set** | 50,000 images |
| **Test Set** | 10,000 images |
| **Validation Split** | 20% of training data → 10,000 validation images |
| **Image Dimensions** | 32 × 32 × 3 (RGB color) |
| **Number of Classes** | 10 |
| **Task Type** | Supervised Multi-class Classification |

**Class Labels:**

| Label | Class | Label | Class |
|:---:|:---:|:---:|:---:|
| 0 | ✈️ Airplane | 5 | 🐶 Dog |
| 1 | 🚗 Automobile | 6 | 🐸 Frog |
| 2 | 🐦 Bird | 7 | 🐴 Horse |
| 3 | 🐱 Cat | 8 | 🚢 Ship |
| 4 | 🦌 Deer | 9 | 🚛 Truck |

**Preprocessing Steps:**

| Step | Operation | Detail |
|---|---|---|
| Normalization | Pixel scaling | `[0, 255]` → `[0.0, 1.0]` (÷ 255.0) |
| Flattening | Reshape | `(32, 32, 3)` → `(3072,)` flat vector |
| Label Encoding | One-hot encoding | `to_categorical(y, 10)` |

---

## 📦 Libraries Used

| Library | Version | Purpose |
|---|---|---|
| `tensorflow` | 2.x | Core deep learning framework |
| `tensorflow.keras` | (included) | High-level model building API |
| `numpy` | ≥ 1.21 | Numerical array operations |
| `matplotlib` | ≥ 3.4 | Training curve visualization & image plotting |
| `python` | 3.12 | Runtime environment |
| `jupyter` | Latest | Interactive development environment |

---

## 📁 Project Structure

```
Deep Learning Project CIFAR-10 Image Classifier/
│
├── Deep_Learning_Project_CIFAR_10_Image_Classifier.ipynb   ← Main Notebook
│
└── README.md                                                ← Project Documentation
```

**Notebook Sections:**

| Section | Title |
|:---:|---|
| 1 | Introduction & Project Goals |
| 2 | Dataset Loading & Preprocessing |
| 3 | Model Architecture Definition |
| 4 | Model Training |
| 5 | Visualization (Accuracy & Loss Curves) |
| 6 | Evaluation & Final Results |

---

## 🏗️ Model Architecture

**Model Type:** Dense Neural Network (MLP — Multilayer Perceptron)
**API:** Keras `Sequential`

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Input Layer         →  3,072-dimensional flat vector (32×32×3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Dense(512, relu)    →  Hidden Layer 1
 BatchNormalization  →  Normalize activations
 Dropout(0.3)        →  Regularization (30% neurons dropped)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Dense(256, relu)    →  Hidden Layer 2
 BatchNormalization  →  Normalize activations
 Dropout(0.3)        →  Regularization (30% neurons dropped)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Dense(128, relu)    →  Hidden Layer 3
 BatchNormalization  →  Normalize activations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Dense(64, relu)     →  Hidden Layer 4
 Dropout(0.2)        →  Regularization (20% neurons dropped)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Dense(10, softmax)  →  Output Layer (10-class probability)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Parameter Summary:**

| Parameter Type | Count | Size |
|---|---|---|
| Total Parameters | 1,750,090 | ~6.68 MB |
| Trainable Parameters | 1,748,298 | ~6.67 MB |
| Non-trainable Parameters | 1,792 | ~7.00 KB |

---

## ⚙️ Training Configuration

| Setting | Value |
|---|---|
| **Optimizer** | Adam |
| **Loss Function** | Categorical Crossentropy |
| **Metrics** | Accuracy |
| **Epochs** | 20 |
| **Batch Size** | 64 |
| **Validation Split** | 0.2 (20%) |

---

## 📈 Results & Performance

**Training Progress (Selected Epochs):**

| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 26.60% | 31.33% | 2.0788 | 1.8908 |
| 5 | 40.43% | 42.20% | 1.6606 | 1.8548 |
| 10 | 44.27% | 44.65% | 1.5661 | 1.8131 |
| 15 | 44.77% | 45.72% | 1.5429 | 1.8303 |
| 20 | 46.50% | 47.70% | 1.4981 | 1.7577 |

**Final Test Set Evaluation:**

| Metric | Value |
|---|---|
| **Test Loss** | 1.5069 |
| **Test Accuracy** | **48.14%** |

**MLP vs CNN Performance Comparison:**

| Architecture | Approach | Expected Accuracy on CIFAR-10 |
|---|---|:---:|
| Dense MLP (this project) | Flattened pixels, no spatial awareness | ~48% |
| Simple CNN | Local feature extraction with convolutions | ~70–80% |
| Deep CNN (ResNet, VGG) | Hierarchical spatial feature learning | ~85–95%+ |

> **Why the gap?** CNNs exploit the spatial locality and translational invariance of image data. A MLP flattens the 2D image into a 1D vector, destroying all spatial structure — every pixel is treated as independent, so the network cannot learn local patterns like edges, textures, or shapes.

---

## 💡 Key Findings

1. **48.14% test accuracy** — the expected ceiling for a fully-connected MLP on CIFAR-10, confirming the theoretical limitation of dense networks on spatially rich image data.
2. **Decreasing training loss** across all 20 epochs confirms the model is learning meaningful signal from the training data.
3. **Fluctuating validation loss** indicates moderate overfitting — a known behavior for dense networks without convolutional inductive biases.
4. **Batch Normalization** stabilizes gradient flow and enables faster convergence across the 4 hidden layers.
5. **Dropout regularization** (rates 0.3, 0.3, 0.2) reduces overfitting and improves generalization to unseen images.
6. **Spatial information loss via flattening** is the primary performance bottleneck — this project validates why CNNs were invented.
7. **The project serves as an essential baseline** — understanding MLP limitations directly motivates the adoption of CNNs for image classification tasks.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow numpy matplotlib jupyter
```

### Run Locally

```bash
# Clone the repository
git clone https://github.com/MH-SHUVO20/Deep-Learning-Project-CIFAR-10-Image-Classifier

# Navigate into the project folder
cd "Deep Learning Project CIFAR-10 Image Classifier"

# Launch Jupyter Notebook
jupyter notebook "Deep_Learning_Project_CIFAR_10_Image_Classifier.ipynb"
```

### Run on Google Colab

> No installation needed — runs entirely in the browser.

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the `.ipynb` file via **File → Upload notebook**
3. Run all cells top to bottom (`Runtime → Run all`)

> The CIFAR-10 dataset is downloaded automatically via `tensorflow.keras.datasets.cifar10` — no manual download required.

---

## 📄 License

This project is licensed under the **MIT License**.

<p align="center">
  <a href="https://github.com/MH-SHUVO20/Deep-Learning-Project-CIFAR-10-Image-Classifier/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge" alt="MIT License"/>
  </a>
</p>

See the [LICENSE](https://github.com/MH-SHUVO20/Deep-Learning-Project-CIFAR-10-Image-Classifier/blob/main/LICENSE) file on GitHub for full details.

---

## 👤 Author

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a237e,50:4527a0,100:7c4dff&height=120&text=MD.%20MEHEDI%20HASAN%20SHUVO&fontSize=26&fontColor=ffffff&animation=fadeIn&fontAlignY=70" width="100%"/>

<br/>

<p align="center">
  <a href="https://github.com/MH-SHUVO20">
    <img src="https://avatars.githubusercontent.com/u/125986989?v=4" width="110px" alt="MH-SHUVO20"/>
  </a>
  <br/>
  <b>MD. MEHEDI HASAN SHUVO</b>
  <br/><br/>
  <a href="https://github.com/MH-SHUVO20">
    <img src="https://img.shields.io/badge/GitHub-MH--SHUVO20-181717?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
  &nbsp;
  <img src="https://komarev.com/ghpvc/?username=MH-SHUVO20&style=for-the-badge&color=7c4dff&label=PROFILE+VIEWS" alt="Profile Views"/>
</p>

<br/>

<table align="center">
  <tr><th>Field</th><th>Details</th></tr>
  <tr><td align="center">🧑 <b>Full Name</b></td><td align="center">MD. MEHEDI HASAN SHUVO</td></tr>
  <tr><td align="center">🐙 <b>GitHub</b></td><td align="center"><a href="https://github.com/MH-SHUVO20">@MH-SHUVO20</a></td></tr>
  <tr><td align="center">🎯 <b>Role</b></td><td align="center">Project Creator & Owner</td></tr>
  <tr><td align="center">🔬 <b>Domain</b></td><td align="center">Deep Learning / Computer Vision</td></tr>
  <tr><td align="center">🛠️ <b>Tools Used</b></td><td align="center">Python · TensorFlow · Keras · NumPy · Matplotlib</td></tr>
  <tr><td align="center">📅 <b>Year</b></td><td align="center">2025</td></tr>
  <tr><td align="center">💻 <b>Platform</b></td><td align="center">Google Colab / Jupyter Notebook</td></tr>
</table>

<br/>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=18&pause=1000&color=7C4DFF&center=true&vCenter=true&width=650&lines=Deep+Learning+%7C+Image+Classification%3BDense+Neural+Network+%7C+CIFAR-10%3BTensorFlow+%7C+Keras+%7C+Batch+Normalization%3BProject+by+MD.+MEHEDI+HASAN+SHUVO" alt="Typing SVG"/>
</p>

<br/>

<p align="center">
  <img src="https://streak-stats.demolab.com?user=MH-SHUVO20&theme=dark&hide_border=true&background=0D1117&ring=7c4dff&fire=ff6b6b&currStreakLabel=7c4dff" alt="GitHub Streak"/>
</p>

<br/>

> 💬 *"Every deep learning journey begins with understanding the limits of the simplest model — and CIFAR-10 with a Dense Network teaches that better than any textbook."*
> — **MD. MEHEDI HASAN SHUVO**

---

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:7c4dff,50:4527a0,100:1a237e&height=100&section=footer" width="100%"/>

<p align="center">
  ⭐ If this project helped you, consider giving it a <strong>star</strong> on GitHub!<br/><br/>
  <strong>Made with ❤️ by MD. MEHEDI HASAN SHUVO — 2025</strong>
</p>
