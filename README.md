# ml-based-orbital-correction_app
# Physics-Informed Orbit Correction

Hybrid GMAT + Transformer orbit correction framework using Ajisai CPF laser-ranging data.

## Features

- CPF ingestion
- GMAT propagation ingestion
- RTN residual learning
- Transformer correction
- 3D orbit visualization
- RMSE metrics
- Upload mode
- Demo mode

## Results

| Metric | Before | After |
|----------|----------|----------|
| Total RMSE | ~7 km | ~50 m |

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
