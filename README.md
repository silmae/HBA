# HyperBlend Algae version 0.2.1a

This is the first release of HyperBlend Algae, which is forked from the original HyperBlend 
project at https://github.com/silmae/hyperblend and modified for photobioreactor design purposes. 
This fork and the orginal one has been developed at [Spectral Laboratory](https://www.jyu.fi/it/en/research/our-laboratories/hsi) of University of Jyväskylä 
by Kimmo Riihiaho (kimmo.a.riihiaho at jyu.fi). 
You can freely use and modify this software under terms of the MIT licence (see LICENCE file). 
If you use the software, please give us credit for our work by citing our published scientific 
papers (instructions below).

As a kind disclaimer: this code is brutally modified from the original using hard-coded values 
and function names that do not really make sense in the photobioreactor context (e.g., rendering 
the reactor scene is done by calling ```render_forest()```). We do not expect to continue developing 
this fork in the future, but instead focus on making the original fork more accomodating for different 
simulation schemes. This fork exist mainly for purposes of reproducing the published results in the 
future if the need arises.

### Table of contents 

  1. [How to cite](#How_to_cite)
  1. [Installing](#Installing)
  1. [Working principle](#Working_principle)
  1. [Usage](#Usage)

##  <a name="How_to_cite"></a> How to cite

If you find our work usefull in your project, please cite us:

### The first HyperBlend paper

Riihiaho, K. A., Rossi, T., and Pölönen, I.: HYPERBLEND: SIMULATING SPECTRAL REFLECTANCE AND TRANSMITTANCE OF LEAF TISSUE WITH BLENDER, ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci., V-3-2022, 471–476, https://doi.org/10.5194/isprs-annals-V-3-2022-471-2022, 2022.

```bibtex
@Article{riihiaho22,
AUTHOR = {Riihiaho, K. A. and Rossi, T. and P\"ol\"onen, I.},
TITLE = {HYPERBLEND: SIMULATING SPECTRAL REFLECTANCE AND TRANSMITTANCE OF LEAF TISSUE WITH BLENDER},
JOURNAL = {ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
VOLUME = {V-3-2022},
YEAR = {2022},
PAGES = {471--476},
URL = {https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-3-2022/471/2022/},
DOI = {10.5194/isprs-annals-V-3-2022-471-2022}
}
```

### The second HyperBlend paper

Riihiaho, K. A., Lind, L., & Pölönen, I. (2023). HyperBlend leaf simulator: improvements on simulation speed, generalizability, and parameterization. Journal of Applied Remote Sensing, 17(3), 038505-038505.

```bibtex
@article{riihiaho2023hyperblend,
  title={HyperBlend leaf simulator: improvements on simulation speed, generalizability, and parameterization},
  author={Riihiaho, Kimmo A and Lind, Leevi and P{\"o}l{\"o}nen, Ilkka},
  journal={Journal of Applied Remote Sensing},
  volume={17},
  number={3},
  pages={038505--038505},
  year={2023},
  publisher={Society of Photo-Optical Instrumentation Engineers}
}
```

### The third (this one) HyperBlend paper

Coming soonish...


## <a name="Installing"></a> Installing

Clone the repository to some location on your machine. Create a python environment by running 
`conda env create -n hb --file hb_env.yml` in yor anaconda command prompt when in project root directory.
Use your favourite IDE for editing and running the code (developed using PyCharm). 
Command line build and run is untested, but it should work as well.

You will also need open-source 3D-modeling and rendering software Blender, which 
you can download and install from ([blender.org](blender.org)). Tests in the paper were run with Blender 4.0.X. 
Other versions may or may not work, depending on the API changes made by Blender. Note that at least 
Blender 3.6 is slower by one order of magnitude in reder time of the algae scenes (tested with the 
validation scene).
Change your Blender executable path to `constants.py`.

## <a name="Working_principle"></a> Working principle

For algae, this is kind of a leaf made out of algae. The paper calls this stage the 
slab simulation.

The measured (or simulated) reflectances and transmittances look like this:  

| wavelength [nm] | reflectance | transmittance
|---|---|---|
|400 | 0.21435 | 0.26547|
|401 | 0.21431 | 0.26540|
|... | ... | ... |

We call this the *target*. Reflectance and transmittance values represent the fraction of 
reflected and transmitted light so both values are separately bound to closed interval [0,1] 
and their sum cannot exceed 1. 

We use a Blender scene with a rectangular box that represents a leaf (```scene_leaf_material.blend```). 
The material of the leaf has four adjustable parameters: absorption particle density, scattering 
particle density, scattering anisotropy, and mix factor. These control how the light is scattered and 
absorbed in the leaf material. HyperBlend will find the values that will produce the same reflectance 
transmittance spectrum that the target has.

For each wavelength in the target, we adjust the leaf material parameters until the modeled 
reflectance and transmittance match the target values. As of version 0.2.0, there are three 
different ways to do this:

  1. Original optimization method (slow but accurate)
  1. Surface fitting method approximates parameter spaces with analytical function surfaces
  1. NN (neural network) does the same as Surface fitting but uses NN instead of analytical functions 

NN is currently the default and recommended method as it is fast (200 x speedup compared to Original) 
and fairly accurate (error increased 2-4 times compared to Original). Surface fitting is as fast as NN 
but not as accurate.


## <a name="Usage"></a> Usage

The entry point of the software is `__main__.py` file. The experimetns run for the results in the 
paper can be found from the main method at `__main__.py`. Do not expect that all of them work without 
tinkering.
