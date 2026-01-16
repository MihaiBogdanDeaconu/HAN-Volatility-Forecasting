# Hierarchical Attention Networks for Multi-Scale Financial Volatility Forecasting (HAN-T)

This repository contains the official PyTorch implementation of the **Hierarchical Attention Network (HAN-T)** for financial volatility forecasting. This architecture explicitly models the multi-scale nature of financial time series by processing high-frequency microstructural data, daily volatility clusters, and weekly trends through dedicated scale-specific encoders fused by a hierarchical attention mechanism.

The model was developed and validated on the **Optiver Realized Volatility Prediction** dataset, achieving state-of-the-art performance against strong baselines like LightGBM and standard Transformers.

## 📂 Repository Structure

```text
.
├── src/
│   ├── features/           # Feature engineering pipeline
│   │   ├── preprocess.py   # Raw data cleaning and aggregation
│   │   ├── neighbors.py    # Nearest Neighbor (Canberra/Mahalanobis) feature construction
│   │   └── transform.py    # Rank normalization, Log scaling, and LDA embeddings
│   ├── models/             # PyTorch model definitions
│   │   ├── encoders.py     # Asymmetric Scale-Specific Encoders
│   │   ├── fuser.py        # Hierarchical Attention Fuser
│   │   └── han.py          # Main HAN-T Model Assembly
│   ├── training/           # Training loops and validation
│   │   ├── train.py        # Main training script
│   │   └── validation.py   # Time-Aware Blocked Cross-Validation implementation
│   └── utils/              # Helper functions (metrics, logging, seeding)
├── notebooks/              # Jupyter notebooks for EDA and rapid prototyping
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

```

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* PyTorch 1.10+ (CUDA recommended for training)
* 16GB+ RAM (due to high-dimensional feature engineering)

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/mihaibogdandeaconu/han-volatility-forecasting.git
cd han-volatility-forecasting
pip install -r requirements.txt

```

### 2. Data Setup

Due to licensing restrictions, the dataset cannot be hosted directly in this repository.

1. Download the dataset from the [Kaggle Optiver Competition](https://www.kaggle.com/c/optiver-realized-volatility-prediction).
2. Extract the files into a local `data/` directory.
3. Ensure your directory looks like this:

```text
data/
├── book_train.parquet/
├── trade_train.parquet/
└── train.csv

```

### 3. Feature Engineering

Run the preprocessing pipeline to generate the multi-scale feature sets. This script extracts microstructural metrics (WAP, spreads), computes nearest-neighbor features, and generates the Short, Mid, and Long-scale input streams.

```bash
python src/features/preprocess.py --input_dir ./data --output_dir ./processed_data

```

> **Note:** This process is computationally intensive and may take several hours depending on your hardware.

### 4. Training & Reproduction

To replicate the results reported in the paper (Report 3), run the main training script. This will execute the **Time-Aware 5-Fold Cross-Validation**.

```bash
python src/training/train.py --data_dir ./processed_data --batch_size 128 --epochs 20 --lr 1e-4

```

The script will save the best model checkpoints to `checkpoints/` and log training metrics (RMSPE) to `logs/`.

## 📊 Results

The HAN-T model achieves superior performance compared to both econometric and machine learning baselines on the Optiver dataset.

| Model | Mean RMSPE | Std Dev |
| --- | --- | --- |
| GARCH(1,1) | 0.2853 | 0.0118 |
| LightGBM | 0.2132 | 0.0053 |
| LightGBM + Optuna | 0.2076 | 0.0044 |
| Flat Transformer | 0.1989 | 0.0041 |
| **HAN-T (Ours)** | **0.1965** | **0.0025** |

## 🛠️ Implementation Details

* **Asymmetric Encoders:**
* **Short-Scale:** 4 layers, 8 heads (captures complex microstructure).
* **Mid-Scale:** 2 layers, 4 heads (captures daily clustering).
* **Long-Scale:** 1 layer, 2 heads (captures weekly trends).


* **Hierarchical Fuser:** A lightweight Transformer that dynamically weighs the importance of each time scale based on the current market context.
* **Validation:** Time-Aware Blocked K-Fold to prevent lookahead bias.

## 📜 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{deaconu2025hierarchical,
  title={Hierarchical Attention Networks for Multi-Scale Financial Volatility Forecasting},
  author={Deaconu, Mihai Bogdan and Pop, Ioan Daniel},
  year={2025}
}

```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

```

---

**Next Step:** Would you like me to generate a `requirements.txt` file based on the libraries mentioned in this README (PyTorch, Pandas for parquet files, etc.) to go along with it?

```
