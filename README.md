# sintel-analysis

Replication files for Sintel.

> S. Alnegheimish, D. Liu, C. Sala, L. Berti-Equille, and K. Veeramachaneni. "Sintel: A Machine Learning Framework to Extract Insights from Signals" SIGMOD 2022. 

## Usage

Experiments were made in **python 3.7**.
To reproduce the analysis made, create a virtual environment and install required packages, then run the script directly. All results will be saved to `./output` directory.

```bash
python -m venv sintel-env
source sintel-env/bin/activate
pip install -r requirements.txt
python analysis.py
```

## Figures

To reproduce figures, refer to `notebooks/analysis.ipynb`.