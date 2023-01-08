This repository contains various scripts to tackle some of the structural biology problems I ran into while completing my doctorate. This is slowly being updated as I go through my work and beautify everything.

## Table of contents

### Notebooks

* **DEERNet:** A Colab notebook that reproduces *most* of the training process and functionality of the [DEERNet neural network](https://www.science.org/doi/10.1126/sciadv.aat5218). The reproduction is carried out in Jax.
* **Fit activity data:** A Colab notebook that fits some activity data using a Michaelis-Menten model in Python using a standard nonlinear curve fit.
* **MDDS:** A Colab notebook and set of accompanying scripts for reproduction of the [Molecular Dynamics of Dummy Spin Labels](https://pubs.acs.org/doi/10.1021/jp311723a) program available on the [CHARMM-GUI web server](https://charmm-gui.org/). Instead of using molecular dynamics force fields, which are costly and can take up to 24 hours for the full 5 nanosecond simulation, this implementation using the [No U-Turn Sampler](http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf), a Hamiltonian Monte Carlo sampler that samples the available space more effectively than classical random-walk Monte Carlo.

### Scripts

* **CM:** A series of scripts that does comparative modeling of a protein into a new conformation. This is intended to be used when the protein of interest is only known in a single conformation, and a different conformation adopted by a homolog is desired. The alignment is carried out using [TM-Align](https://zhanggroup.org/TM-align/) and the sequence of interest is threaded using this alignment onto the target structure using the Rosetta application *partial_thread*. Then, RosettaCM (more specifically [HybridizeMover](https://www.cell.com/structure/fulltext/S0969-2126(13)00297-9), implemented in [RosettaScripts](http://www.meilerlab.org/index.php/publications/showPublication/pub_id/98)) is applied to optimize backbone geometry, close loops, and add side chains. This script uses multiprocessing, but for best results, you should run on a computing cluster and generate hundreds to thousands of models.
* **TM-PCA:** Scripts to align and interpret a set of models. Alignment is carried out using [TM-Align](https://zhanggroup.org/TM-align/), while principal component analysis is carried out using [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html). These scripts make extensive use of [BioPython](https://biopython.org/).
* **Report residue movement:** Script to align two conformations of the same protein and report on the movement of each alpha carbon. Alignment is carried out using [TM-Align](https://zhanggroup.org/TM-align/). This script uses [BioPython](https://biopython.org/).

### Extras

* **AAT_OmegaFold:** A series of protein structural models generated with [OmegaFold](https://www.biorxiv.org/content/10.1101/2022.07.21.500999v1) as part of a public demonstration (see [this Twitter thread](https://twitter.com/DdelAlamo/status/1571483783672791042)).
