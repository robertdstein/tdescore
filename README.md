# tdescore

**Disclaimer: This code is provided in an open source format in case pieces are helpful to others. 
However, the ZTF classifications used to train tdescore have not yet been released! 
You can (and are strongly encouraged to) use tdescore for your own classifier projects, 
but without access to internal ZTF data you cannot directly reproduce the original tdescore analysis.**

# Install Instructions

tdescore is a python package. We recommend using conda to install it.

```commandline
conda create -n tdescore python=3.11
conda activate tdescore
git clone git@github.com:robertdstein/tdescore.git
pip install -e tdescore
```
(Python 3.12 is not yet supported as of 2023-11-22, but is expected soon).

## Sfdmap

You will also need to install the sfdmap2 package, and sfdmap files.

See instructions at
https://github.com/AmpelAstro/sfdmap2

# Usage

tdescore is modular.

First you need raw data. ZTF collaboration members can use Ampel to download ZTF lightcurves:

```commandline
python -m tdescore.raw
```

However, any raw data in the appropriate jsonnstyle would work. 
It does not need to be from ZTF! An example is provided under sample_data.

Next, you should collate the additional data you want to use for classification. 
You can run these commands in any order, and omit steps you do not want.

* For downloading cross-matched data from public catalogs:
    ```commandline
    python -m tdescore.download
    ```

* For analysing lightcurves with gaussian processes: 
    ```commandline
    python -m tdescore.lightcurve
    ```
  
