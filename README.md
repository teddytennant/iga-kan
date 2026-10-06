IGA-KAN is a JAX implementation of the local ridge post-process from arXiv:2610.06348.
pip install -e .
python3 -m pytest
The fit stacks the strong-form residual, the isogeometric data, and the Dirichlet rows on a tensor-product mesh, then blends the local models with the hat functions. Tests use dloc=3 and Q=4 on the unit square, with nodal interpolation in place of the Galerkin solve. They do not match the paper's L2 tables.
