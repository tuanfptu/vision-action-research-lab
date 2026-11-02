# OpenArm notebook workflow

Use the new .venv-gpu kernel only. Executable source remains the package and scripts.

```python
from va_lab.datasets.core import read_arrays
from va_lab.evaluation.visualize import inspect
from IPython.display import Image, display
arrays=read_arrays("data/openarm.npz")
print({k:v.shape for k,v in arrays.items()})
inspect("data/openarm.npz","runs/openarm_observations.png")
display(Image("runs/openarm_observations.png"))
```

Every plot includes selected named cameras and all 16 joints. Use configured camera selection,
never slice action vectors to reuse a different embodiment. Do not use old cached notebooks.
