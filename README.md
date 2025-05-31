#CharCNN Fine-Tuning for Fake News Detection
This project implements a character-level Convolutional Neural Network (CharCNN) for detecting fake news. The model is initially trained on a general dataset and then fine-tuned using the ISOT dataset. All training and evaluation steps are logged using Weights & Biases (wandb).
---
##Features
- Character-level input encoding (text-to-sequence)
- Deep CharCNN architecture
- Initial training using Fake.csv and True.csv
- Fine-tuning using ISOT_Fake.csv and ISOT_True.csv
- Metric logging via wandb
- Confusion matrix and classification report visualization
---
##Project Structure
```
charcnn_finetune/
├── charcnn_finetune.ipynb        # Main training and fine-tune notebook
├── charcnn_finetune_colab.ipynb  # Google Colab compatible version
├── .gitignore                    # Excludes data/, venv/, wandb/
├── README.md                     # This file
├── base_model.pt                 # Model saved after initial training
└── data/
    ├── Fake.csv
    ├── True.csv
    ├── ISOT_Fake.csv
    ├── ISOT_True.csv
```
---
##Datasets Used
- `Fake.csv`, `True.csv` — Kaggle Fake News Dataset
- `ISOT_Fake.csv`, `ISOT_True.csv` — ISOT Dataset
- Optionally: `train.tsv` — LIAR Dataset
---
##Monitoring and Visualization
All metrics and training logs are integrated with [wandb.ai](https://wandb.ai):
- Training loss and accuracy
- Test performance
- Confusion matrix visualizations
---
##Local Setup
```bash
python3 -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt
```
---
##Dependencies
```bash
pip install torch pandas scikit-learn seaborn matplotlib wandb
```
---
##How to Use
Run the notebook step-by-step:
1. Open `charcnn_finetune.ipynb`
2. Run training and evaluation cells
3. Log into wandb when prompted
   
For Colab:
- Upload `charcnn_finetune_colab.ipynb` to [Google Colab](https://colab.research.google.com)
- Mount Google Drive to access dataset files
- Start training and monitoring

##License
MIT License © 2025
