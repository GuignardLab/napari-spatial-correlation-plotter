# :herb: napari-spatial-correlation-plotter

[![License MIT](https://img.shields.io/pypi/l/napari-spatial-correlation-plotter.svg?color=green)](https://github.com/jules-vanaret/napari-spatial-correlation-plotter/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-spatial-correlation-plotter.svg?color=green)](https://pypi.org/project/napari-spatial-correlation-plotter)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-spatial-correlation-plotter.svg?color=green)](https://python.org)
[![tests](https://github.com/jules-vanaret/napari-spatial-correlation-plotter/workflows/tests/badge.svg)](https://github.com/jules-vanaret/napari-spatial-correlation-plotter/actions)
[![codecov](https://codecov.io/gh/jules-vanaret/napari-spatial-correlation-plotter/branch/main/graph/badge.svg)](https://codecov.io/gh/jules-vanaret/napari-spatial-correlation-plotter)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-spatial-correlation-plotter)](https://napari-hub.org/plugins/napari-spatial-correlation-plotter)

<img src="https://github.com/GuignardLab/tapenade/blob/Packaging/imgs/tapenade3.png" width="100">

A plugin to dynamically interact with the spatial correlation heatmap obtained by comparing two continuous fields of biophysical properties in 3D tissues.

`napari-spatial-correlation-plotter` is a [napari] plugin that is part of the [Tapenade](https://github.com/GuignardLab/tapenade) project. Tapenade is a tool for the analysis of dense 3D tissues acquired with deep imaging microscopy. It is designed to be user-friendly and to provide a comprehensive analysis of the data.

If you use this plugin for your research, please [cite us](https://github.com/GuignardLab/tapenade/blob/main/README.md#how-to-cite).

## Overview

While working with large and dense 3D and 3D+time gastruloid datasets, we found that being able to visualise and interact with the data dynamically greatly helped processing it.
During the pre-processing stage, dynamical exploration and interaction led to faster tuning of the parameters by allowing direct visual feedback, and gave key biophysical insight during the analysis stage.

This plugins allows the user to analyse the spatial correlations of two 3D fields loaded in Napari (e.g two fluorescent markers). The user can dynamically vary the analysis length scale, which corresponds to the standard deviation of the Gaussian kernel used for smoothing the 3D fields. 
If a layer of segmented nuclei instances is additionally specified, the histogram is constructed by binning values at the nuclei level (each point corresponds to an individual nucleus). Otherwise, individual voxel values are used.
The user can dynamically interact with the correlation heatmap by manually selecting a region in the plot. The corresponding cells (or voxels) that contributed to the region's statistics will be displayed in 3D on an independant Napari layer for the user to interact with and gain biological insight.

## Installation

The plugin obviously requires [napari] to run. If you don't have it yet, follow the instructions [here](https://napari.org/stable/tutorials/fundamentals/installation.html).

The simplest way to install `napari-spatial-correlation-plotter` is via the [napari] plugin manager. Open Napari, go to `Plugins > Install/Uninstall Packages...` and search for `napari-spatial-correlation-plotter`. Click on the install button and you are ready to go!

You can also install `napari-spatial-correlation-plotter` via [pip]:

    pip install napari-spatial-correlation-plotter

To install latest development version :

    pip install git+https://github.com/jules-vanaret/napari-spatial-correlation-plotter.git

## Usage

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-spatial-correlation-plotter" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/jules-vanaret/napari-spatial-correlation-plotter/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
