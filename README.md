# sintel-analysis

Replication files for Sintel.

> S. Alnegheimish, D. Liu, C. Sala, L. Berti-Equille, and K. Veeramachaneni. "Sintel: An Overarching Ecosystem for End-to-End Time Series Anomaly Detection" SIGMOD 2022. 

## Usage

Create a virtual environment and install required packages, then run the script directly. All results will be saved to `./output` directory.

```bash
python -m venv sintel-env
source sintel-env/bin/activate
pip install -r requirements.txt
python analysis.py
```