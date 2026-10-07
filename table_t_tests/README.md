# Small Samples on Trial

A small Streamlit app for screen-recording two reproducible t-test demos. Each tab lets you choose the example measurements or enter your own, select a directional alternative hypothesis and significance level, and see the hypotheses, calculation, t distribution, decision, and downloadable Python code update together.

## Run on Windows (PowerShell)

From this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens in your browser. Select each tab while recording. The app includes voice-over cues and lower-third captions. Enter measurements as comma- or space-separated numbers; each sample needs at least two finite numeric values.

For a narration script and screen-by-screen recording cues, see [SPEAKER_NOTES.md](SPEAKER_NOTES.md).

## Reproducibility notes

- The six tablet weights are the exact fixed observations from the demo. The brief labels the scenario `seed = 8`; because its observations are supplied explicitly, the seed is not needed to reproduce the calculations.
- Supplier measurements are explicit fixed values, with five observations per supplier and the stated means. The two-sample t-test uses the equal-variance Student test.
- The incorrect z comparison treats the observed sample SD as if it were a known population SD. Its p-value is approximately 2.5 million times smaller than the t-test p-value for these measurements, not merely a thousand times smaller.

`statistics_demo.py` contains the calculations used in the app. The downloadable code on each tab independently reproduces its result using NumPy and SciPy.
