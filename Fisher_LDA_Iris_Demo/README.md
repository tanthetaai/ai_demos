# Fisher's Linear Discriminant Analysis on Iris

This standalone demo uses Fisher's Iris measurements to compare one-feature
classifiers with a linear discriminant score, then extends the example to all
three species.

## Setup

From this folder, create and activate a virtual environment, then install the
demo's dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open `Fisher_LDA_Iris_Demo.ipynb` in Jupyter or VS Code and run all cells.
The Iris dataset is bundled with scikit-learn; no data download is needed.
