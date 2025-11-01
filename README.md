
# Install

```bash
conda create -n jaxtutorial python==3.12
conda activate jaxtutorial
pip install -U "jax[cuda12]"
pip install --quiet flax
pip install --quiet optax
pip3 install torch torchvision # cuda==12.8
pip install -r requirements.txt
```



# 资料来源

- [UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html)