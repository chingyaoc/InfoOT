# InfoOT: Information Maximizing Optimal Transport

Optimal transport aligns samples across distributions by minimizing the transportation cost between them, e.g., the geometric distances. Yet, it ignores coherence structure in the data such as clusters, does not handle outliers well, and cannot integrate new data points. To address these drawbacks, we propose InfoOT, an information-theoretic extension of optimal transport that maximizes the mutual information between domains while minimizing geometric distances. The resulting objective can still be formulated as a (generalized) optimal transport problem, and can be efficiently solved by projected gradient descent. This formulation yields a new projection method that is robust to outliers and generalizes to unseen samples. 


**InfoOT: Information Maximizing Optimal Transport** [[paper]](https://arxiv.org/abs/2007.00224)
<br/>
[Ching-Yao Chuang](https://chingyaoc.github.io/), 
[Stefanie Jegelka](https://people.csail.mit.edu/stefje/), and
[David Alvarez-Melis](https://dmelis.github.io/)
<br/>

## Prerequisites
- Python 3.7 
- POT
- tqdm
- scikit-learn

## Usage Examples
The code for InfoOT lie in `infoot.py`. For instance, the following code solves fused InfoOT given two data matrices:
```python
# Xs: [n, d]
# Xt: [m, d]
from infoot import FusedInfoOT

ot = FusedInfoOT(Xs, Xt, h=0.5, reg=1.)
P = ot.solve()
```
If the source label is given, one can use it to refine the source pairwise distance as follows:
```python
# Ys: [n]
ot = FusedInfoOT(Xs, Xt, Ys=Ys, h=0.5, reg=1.)
P = ot.solve()
```

Many applications of optimal transport involve mapping source points to a target domain. One can perform either barycentric or conditional projection with the following code. Note that the conditional projection can generalize to unseen samples.
```python
# project the source onto target
ProjX1 = ot.project(Xs, method='barycentric')
ProjX2 = ot.project(Xs, method='conditional')
```

For aligning domains whose supports lie in different metric spaces, e.g., supports with different modalities or dimensionality, one can simply adopt the standar InfoOT:
```python
# Xs: [n, d1]
# Xt: [m, d2]
# d1 != d2
from infoot import InfoOT

ot = InfoOT(Xs, Xt, h=0.5, reg=0.05)
P = ot.solve()
```


Other useful functions for computing kernels, the gradient w.r.t. mutual information, projection can also be found in `infoot.py`.

## Domain Adaptation
Download the DeCAF feature for Office-Caltech dataset [here](https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md#office+caltech) and place the data in directory `decaf6`. The following script reproduces the result with barycentric and conditional projection.
```
python domain_adapt.py --src caltech --tgt dslr
```

## Cross-Domain Retrieval
We will use the same data from the domain adaptation experiment. The following script reproduces the result with the conditional score.
```
python retrieval.py --src caltech --tgt dslr
```


## Citation

```
@article{chuang2022info,
  title={InfoOT: Information Maximizing Optimal Transport},
  author={Chuang, Ching-Yao and Jegelka, Stefanie and Alvarez-Melis, David},
  journal={arXiv preprint arXiv:????.?????},
  year={2022}
}
```
