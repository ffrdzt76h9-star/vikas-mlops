# Mobile Phone Price-Tier Classification (MLOps project)

Classifies phones as **Budget (0)** or **Premium (1)**. The label is derived from RAM, battery,
resolution and internal memory (+ noise) in `src/target.py`, exactly as in the original notebook.

Run everything from the project root:

```bash
pip install -r requirements.txt
python pipelines/run_lab3_baseline.py   # preprocess -> train 8 models -> evaluate best
python pipelines/run_lab4_tracking.py   # MLflow experiment tracking + reproducibility check
python pipelines/run_lab5_pipeline.py   # schema validation -> sklearn pipeline -> output validation
python pipelines/run_lab6_registry.py   # register model -> champion/challenger -> deployment report
```

Labs 4 and 6 need `mlflow`; Lab 5 needs `pandera`. Run Lab 5 before Lab 6 (Lab 6 logs `models/preprocessor.pkl`).
The exploratory notebook is in `notebooks/`; production logic lives in `src/` and `pipelines/`.
