# tdescore

**Operation something-for-nothing.**

tdescore is modular.

First you need to download raw data:

```commandline
python -m tdescore.raw
```

Next, you should collate the additional data you want to use for classification. 
You can run these commands in any order, and omit steps you do not want.

* For downloading crossmatched data:
    ```commandline
    python -m tdescore.download
    ```

* For analysing lightcurves with gaussian processes: 
    ```commandline
    python -m tdescore.lightcurve
    ```
