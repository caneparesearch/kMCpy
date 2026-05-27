# Analyze Results

The first analysis step is to check that the run reached a diffusive regime and
that the reported transport quantities are stable enough for the scientific
question.

## Load The Result Table

```python
import pandas as pd

results = pd.read_csv("results_NASICON_298K.csv.gz")
print(results.tail())
```

Common columns are:

- `time`,
- `msd`,
- `jump_diffusivity`,
- `tracer_diffusivity`,
- `conductivity`,
- `havens_ratio`,
- `correlation_factor`.

## Check Diffusive Behavior

Plot MSD against time:

```python
import matplotlib.pyplot as plt

plt.plot(results["time"], results["msd"])
plt.xlabel("time / s")
plt.ylabel("MSD / Angstrom^2")
plt.show()
```

For reliable diffusivity, the production part of the trajectory should show an
approximately linear MSD-time relation. If the curve is noisy or dominated by
early transients, increase equilibration, production passes, or the number of
independent runs.

## Compare Transport Quantities

Tracer diffusivity follows individual ion motion. Jump diffusivity follows
collective charge motion and is used for conductivity. The Haven ratio compares
these two diffusivities and is sensitive to correlated motion.

Use the last part of the trajectory or averages over independent seeds rather
than a single early sample.

## Reproducibility Checklist

Record:

- kMCpy version,
- structure file or structure hash,
- `site_mapping`,
- event library,
- model file,
- initial occupations,
- temperature and attempt frequency,
- equilibration and production passes,
- random seed strategy.

The `Configuration` and model files are designed to make this information easy
to keep with the results.
