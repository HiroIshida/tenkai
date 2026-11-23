# Porting provenance

| Item | Upstream repository | License | Files in tenkai | Porting method | AI model |
| --- | --- | --- | --- | --- | --- |
| Shortcut algorithm | [ompl/ompl](https://github.com/ompl/ompl) | [BSD-3-Clause](https://github.com/ompl/ompl/blob/1.6.0/LICENSE) | [shortcut.cuh](src/tenkai/templates/rrt/shortcut.cuh), [shortcut.py](src/tenkai/shortcut.py) | Understand the [random shortcut algorithm](https://github.com/ompl/ompl/blob/1.6.0/src/ompl/geometric/src/PathSimplifier.cpp#L187) → reimplement in CUDA/HIP; carry over parameter meanings and default behavior. | GPT-6 astra |
