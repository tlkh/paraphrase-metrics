# MRPC Dataset Info

MRPC can be downloaded from: https://www.microsoft.com/en-us/download/details.aspx?id=52398

If you are on a Linux-based system, you can do the following to extract the dataset:

```shell
apt install msitools
msiextract <path to MSI file>
```

If you are on macOS, you can install `msitools` via `brew`. 

The two files needed are:

* `msr_paraphrase_train.txt`
* `msr_paraphrase_test.txt`


Run `mrpc_converter.ipynb` to inspect, convert and process the text files into CSV files in this directory.

