0.0.18 (2025-08-06)
===================

Breaking Changes
----------------

- Where flows allowed an input file to be provided as a ``Callable`` that returned the required file, those flows now expect an object subclassing `DataLoader`. (`#538 <https://github.com/punch-mission/punchbowl/pull/538>`__)
- The L1 flow has been split into 'early' and 'late' phases. Early writes an X{MZPR} file and, for clear inputs, a QR file. Late writes a CR or P{MZP}. (`#554 <https://github.com/punch-mission/punchbowl/pull/554>`__)
- In stray light subtraction, mismatches in observatory or polarization state are now an error, not a warning. (`#554 <https://github.com/punch-mission/punchbowl/pull/554>`__)


New Features
------------

- Stray light models have their date_obs set to the provided reference time and require much less RAM to generate. (`#538 <https://github.com/punch-mission/punchbowl/pull/538>`__)
- Implements second order square root decoding table, using the technique from DeForest et al 2022. (`#550 <https://github.com/punch-mission/punchbowl/pull/550>`__)
- Stray light models can be interpolated or extrapolated to the image time. It will use either DATE-AVG, DATE-BEG, or DATE-END of the stray light models as the reference time for the models, so that in the quickpunch case, a 6-hour isn't extrapolated for days. (`#554 <https://github.com/punch-mission/punchbowl/pull/554>`__)
- Stray light models are written with DATE-OBS set to the reference time, DATE-AVG  set to the mean of the input files' DATE-OBS, and DATE-{BEG,END} set to the earliest/latest input DATE-OBS. (`#554 <https://github.com/punch-mission/punchbowl/pull/554>`__)


Bug Fixes
---------

- Skips PCA testing on macOS and improves git handling of test temporary sample files. (`#546 <https://github.com/punch-mission/punchbowl/pull/546>`__)
- The PSF dilation was too small and resulting in artifacts. This expands it from 1 to 3 dilations. (`#547 <https://github.com/punch-mission/punchbowl/pull/547>`__)
- Provenance wasn't getting populated. This makes sure it's fully filled for all products. (`#551 <https://github.com/punch-mission/punchbowl/pull/551>`__)
- NFI pointing was consistently failing. This improves it by changing the alignment search parameters and allowing astrometry.net to fail by falling back on the spacecraft WCS. (`#557 <https://github.com/punch-mission/punchbowl/pull/557>`__)


Documentation
-------------

- Improves the documentation about file versions. (`#553 <https://github.com/punch-mission/punchbowl/pull/553>`__)
- Updates data query example notebook to streamline download and execution. (`#556 <https://github.com/punch-mission/punchbowl/pull/556>`__)


Internal Changes
----------------

- The ``on_completion`` debugging hook is only set if debugging is enabled, avoiding log messages after every Task run when debugging isn't enabled. (`#538 <https://github.com/punch-mission/punchbowl/pull/538>`__)
- The filler WCS written in stray light models is now a valid, though arbitrary, WCS. (`#538 <https://github.com/punch-mission/punchbowl/pull/538>`__)
- Turns off ruff's EM101 check. (`#558 <https://github.com/punch-mission/punchbowl/pull/558>`__)
- Speedups to alignment and to de-spiking with many flows running at once. (`#559 <https://github.com/punch-mission/punchbowl/pull/559>`__)
- Disables Prefect task result caching (`#560 <https://github.com/punch-mission/punchbowl/pull/560>`__)
