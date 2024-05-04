
![shearmapmpmaps](https://github.com/LaboratoryOfPlasmaPhysics/mpmaps/assets/3200931/ca8b579e-0435-417b-aeab-0b1306311ecb)

# Magnetopause Maps

[![PyPI](https://img.shields.io/pypi/v/mpmaps)](https://pypi.python.org/pypi/mpmaps)
[![Tests](https://github.com/LaboratoryOfPlasmaPhysics/mpmaps/actions/workflows/test_main.yml/badge.svg)](https://github.com/LaboratoryOfPlasmaPhysics/mpmaps/actions/workflows/test_main.yml)
[![Documentation Status](https://readthedocs.org/projects/mpmaps/badge/?version=latest)](https://mpmaps.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://codecov.io/gh/LaboratoryOfPlasmaPhysics/mpmaps/branch/main/graph/badge.svg?branch=main)](https://codecov.io/gh/LaboratoryOfPlasmaPhysics/mpmaps/branch/main)

MMPaps is a package that allows to compute and plot maps of various physical quantities onto the
Earth Magnetopause. Maps are computed from in situ measurements only, following the methodology of
Michotte de Welle et al. 2004


```
@article{MichottedeWelle2024,
title={Global environmental constraints on magnetic reconnection at the magnetopause from in-situ measurements},
DOI={10.22541/essoar.170808382.29449499/v1},
journal={ESS Open Archive},
author={Michotte de Welle, B and Aunai, N and Lavraud, B. and Nguyen, G. and Génot, V and Ghisalberti, A. and Jeandet, A. and Smets, R.},
year={2024}}
```


* Free software: GNU General Public License v3
* Documentation: https://mpmaps.readthedocs.io.

## Features

### compute a shear angle map
```python
import mpmaps
mpm = mpmaps.MPMap()

# set IMF parameters
mpm.set_parameters(tilt=12, clock=127, cone = 22, bimf=5)

# compute and retrieve shear angle values for
# parameters set above
shear_map_values = mpm.shear_angle()

# now let's make a map for northward IMF
# other IMF params are default ones
mpm_north = mpm.MPMap(clock=45)
shear = mpm_north.shear_angle()

```

### compute reconnection rate map


```python
import mpmaps
mpm = mpmaps.MPMap()

# set IMF parameters
mpm.set_parameters(tilt=12, clock=127, cone = 22, bimf=5)

# compute and retrieve rate values for
# parameters set above
# by default the reconnection components are computed
# so that the X line maximizes the Cassak-Shay scaliing law
# (rec_angle='rate')
rate_values = mpm.reconnection_rate()

#now same map but with X line locally aligned with the bisection
rate_values_bisec = mpm.reconnection_rate(rec_angle='bisection')

```


### compute the current density map

```python

import mpmaps
mpm = mpmaps.MPMap()

# set IMF parameters
mpm.set_parameters(tilt=12, clock=127, cone = 22, bimf=5)

values = mpm.current_density()


# default magnetopause thickness is 800km
# but can be changed
mpm.set_parameters(mp_thick=800)
values = mpm.current_density()
```



## Credits

We acknowledge support from:

-  The space plasma team of the [Laboratory of Plasma Physics](https://www.lpp.polytechnique.fr)

