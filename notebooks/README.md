# Notebook workflow

The executable source of truth is `va-lab` and the scripts, so notebook state is never
required for reproducibility. In a fresh notebook kernel pointing to `.venv/bin/python`,
use the following cells to inspect real observations; do not use a previous project's kernel.

```python
from va_lab.datasets.core import read_arrays
from va_lab.evaluation.visualize import inspect
from IPython.display import Image, display
arrays = read_arrays("data/synthetic.npz")
print({k: v.shape for k, v in arrays.items()})
inspect("data/synthetic.npz", "runs/observations.png")
display(Image("runs/observations.png"))
```

No notebook-only training implementation is maintained.
