# MEDIDA for QG

#### [[project website]](http://pedram.rice.edu/team/)
<img src="docs/MEDIDA_QG_Schematic_Steps.png" width="350">
<img src="docs/schematic_interp.png" width="350">

## Table of contents
* [Introduction](#Introduction)
* [Requirements](#Requirements)
* [Experiments](#Experiments)
    * [Case 1](#Case-1)
* [Citation](#Citation)
* [References](#References)

## Introduction
Key points
<ul>
<li>Model error discovery with interpretability and data assimilation (<a href="https://github.com/envfluids/MEDIDA">MEDIDA</a>)[1]* is scaled
up to geostrophic turbulence and sparse observations</li>
<li>Naive use of neural nets (NNs) as interpolator does not capture small scales due to
spectral bias, failing discoveries of closed-form errors</li>
<li>Reducing this bias using random Fourier features enables NNs to represent the full
range of scales, leading to successful error discoveries</li>
</ul>

## Requirements
<!-- These are examples,
	add or remove as appropriate -->

- python 3.6
	- [scipy](https://pypi.org/project/scipy/)
	- [numpy](https://pypi.org/project/numpy/)
- [Pytroch](https://pytorch.org/docs/1.11/)
- [RFF in Pytroch](https://github.com/jmclong/random-fourier-features-pytorch)

## Experiments
### Case 1
Case 1 is disscused here [Case 1 Location](./experiments/QG) 


Python code

```bash
will be updated
```

## Citation
- [Mojgani, R.](https://www.rmojgani.com), [Chattopadhyay, A.](https://scholar.google.com/citations?user=wtHkCRIAAAAJ&hl=en), and [Hassanzadeh, P.
](https://scholar.google.com/citations?user=o3_eO6EAAAAJ&hl=en),
[**Interpretable structural model error discovery from sparse assimilation increments using spectral bias-reduced neural networks: A quasi-geostrophic turbulence test case**](https://arxiv.org/abs/2309.13211), (2023).([url]([https://arxiv.org/abs/2309.13211](https://arxiv.org/abs/2309.13211)))<details><summary>BibTeX</summary><pre>@misc{mojgani2023interpretable,
      title={Interpretable structural model error discovery from sparse assimilation increments using spectral bias-reduced neural networks: {A} quasi-geostrophic turbulence test case}, 
      author={Rambod Mojgani and Ashesh Chattopadhyay and Pedram Hassanzadeh},
      year={2023},
      eprint={2309.13211},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph}
}</pre></details>


## References

- \[1\] [Mojgani, R.](https://www.rmojgani.com), [Chattopadhyay, A.](https://scholar.google.com/citations?user=wtHkCRIAAAAJ&hl=en), and [Hassanzadeh, P.
](https://scholar.google.com/citations?user=o3_eO6EAAAAJ&hl=en),
[**Closed-form discovery of structural errors in models of chaotic systems by integrating Bayesian sparse regression and data assimilation.**](https://doi.org/10.1063/5.0091282), Chaos 32, 061105 (2022) 
arXiv:2110.00546.
([Download](https://aip.scitation.org/doi/pdf/10.1063/5.0091282))<details><summary>BibTeX</summary><pre>
@article{Mojgani_Chaos_2022,
author = {Mojgani,Rambod  and Chattopadhyay,Ashesh  and Hassanzadeh,Pedram },
title = {Discovery of interpretable structural model errors by combining {B}ayesian sparse regression and data assimilation: {A} chaotic {K}uramoto–{S}ivashinsky test case},
journal = {Chaos: {A}n Interdisciplinary Journal of Nonlinear Science},
volume = {32},
number = {6},
pages = {061105},
year = {2022},
doi = {10.1063/5.0091282},
URL = {https://doi.org/10.1063/5.0091282},
eprint = {arXiv:2110.00546}
}</pre></details>
