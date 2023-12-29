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

## Introduction
Key points
<ul>
<li>Model error discovery with interpretability and data assimilation ( [MEDIDA](https://github.com/envfluids/MEDIDA) ) is scaled
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

## Experiments
### Case 1
Case 1 is disscused here [Case 1 Location](./experiments/QG) 


Python code

```bash
will be updated
```
## Citation
- Rambod Mojgani, Ashesh Chattopadhyay, Pedram Hassanzadeh",Interpretable structural model error discovery from sparse assimilation increments using spectral bias-reduced neural networks: A quasi-geostrophic turbulence test case, (2023).([url]([https://arxiv.org/abs/2309.13211](https://arxiv.org/abs/2309.13211)))<details><summary>BibTeX</summary><pre>@misc{mojgani2023interpretable,
      title={Interpretable structural model error discovery from sparse assimilation increments using spectral bias-reduced neural networks: {A} quasi-geostrophic turbulence test case}, 
      author={Rambod Mojgani and Ashesh Chattopadhyay and Pedram Hassanzadeh},
      year={2023},
      eprint={2309.13211},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph}
}</pre></details>



