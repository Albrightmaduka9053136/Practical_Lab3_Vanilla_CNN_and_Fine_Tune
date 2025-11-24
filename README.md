# Practical Lab 3  Vanilla CNN and Fine-Tune VGG16

Maduka Albright Ifechukwude (Student ID: 9053136)

**Summary**

This repository contains a Jupyter notebook that implements and compares two convolutional neural network approaches for the Dogs vs Cats classification task on a reduced dataset (~5,000 images): a Vanilla CNN trained from scratch and a transfer-learning approach using VGG16 (feature extraction + fine-tuning).

**Repository structure**

- `Albright_9053136_Practical_Lab3_Vanilla_CNN_and_Fine_Tune.ipynb`  main notebook with code, figures, and narrative.
- `data/`  expected dataset root (not included). See "Data layout" below.
- `models_lab3/`  saved checkpoints and model weights produced while training.
- `requirements.txt`  Python package pins used for the project.

**Key results (from the notebook)**

- Vanilla CNN test accuracy: ~0.746
- VGG16 fine-tuned test accuracy: ~0.9725

**Environment & Dependencies**

Recommended Python environment (example):

- Python 3.10+
- TensorFlow 2.15
- numpy
- matplotlib
- scikit-learn
- jupyter

Install example (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If you want to install manually (exact TF version used in the notebook):

```powershell
pip install tensorflow==2.15 numpy matplotlib scikit-learn jupyter
```

**Data layout**

Place the dataset under the `data/` folder using the following expected layout (same as used in the notebook):

```
data/kaggle_dogs_vs_cats_small/
  train/
    cats/
    dogs/
  validation/
    cats/
    dogs/
  test/
    cats/
    dogs/
```

Note: The dataset is not committed to the repository due to size. The notebook uses `tf.keras.utils.image_dataset_from_directory` to load these folders.

**Notebook: usage**

1. Open `Albright_9053136_Practical_Lab3_Vanilla_CNN_and_Fine_Tune.ipynb` in Jupyter or VS Code.
2. Ensure the `data/` folder is present and matches the structure above.
3. Start the environment and run cells in order. Recommended to run on GPU if available.

Quick run (PowerShell):

```powershell
# Activate the venv
.\.venv\Scripts\Activate.ps1
# Start Jupyter Notebook
jupyter notebook Albright_9053136_Practical_Lab3_Vanilla_CNN_and_Fine_Tune.ipynb
```

**Notebook outline / sections**

1. Setup and imports (random seed and reproducibility settings)
2. Data loading and EDA (class counts, sample images)
3. Data augmentation pipeline using Keras preprocessing layers
4. Vanilla CNN: architecture, training, and evaluation
5. VGG16 transfer learning: feature extraction and fine-tuning
6. Callbacks: `ModelCheckpoint`, `EarlyStopping` and saving best weights
7. Evaluation: accuracy, confusion matrix, classification report, PR curve
8. Error analysis: visualizing misclassified examples

**Model & training details**

- Vanilla CNN: 3 convolutional blocks (32  64  128 filters), pooling, Dense(128) head, Dropout(0.5). Optimizer: Adam (1e-4). Loss: `binary_crossentropy`.
- VGG16-based: `VGG16(include_top=False, weights='imagenet')` base, `GlobalAveragePooling2D`, Dense(256) + Dropout(0.5), Dense(1, sigmoid) head. Feature-extraction training uses Adam(1e-4); fine-tuning uses Adam(1e-5).

**Notes & tips**

- Training on GPU will speed up experiments substantially.
- If your dataset is larger/different, adjust `batch_size` and training epochs accordingly.
- Check `models_lab3/` after training for saved checkpoints and the best model weights.

**Contact / Author**

Maduka Albright Ifechukwude  Student ID: 9053136

---

If you'd like, I can also:

- Add badges (e.g., Python / TensorFlow) to the top of this README
- Add a short `CONTRIBUTING.md` or a script to download/preprocess the dataset
