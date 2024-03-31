# PLMAlign

This is the implement of <b>PLMAlign</b>, an alignment tool in "PLMSearch: Protein language model powers accurate and fast sequence search for remote homology". PLMAlign takes amino acid-level embeddings as input to obtain specific alignments and corresponding alignment scores.

Specifically, PLMAlign can achieve <b>local</b> and <b>global</b> alignment. The specific algorithm and parameters are similar to the [SW](https://www.ebi.ac.uk/Tools/psa/emboss_water/) and [NW](https://www.ebi.ac.uk/Tools/psa/emboss_needle/) algorithms implemented by [EMBL-EBI](https://www.ebi.ac.uk/). However, by converting a fixed substitution matrix into similarity calculated by the dot product of amino acid-level embeddings, PLMAlign is able to capture deep evolutionary information and perform better on remote homology protein pairs.

<div align=center><img src="example/figure/framework.png" width="100%" height="100%"/></div>

## Quick links

* [Webserver](#webserver)
* [Requirements](#requirements)
* [Data preparation](#data-preparation)
* [Reproduce all our experiments](#main)
* [Build PLMSearch locally](#pipeline)
* [Citation](#citation)

## Webserver
<span id="webserver"></span>

PLMAlign   web server : [dmiip.sjtu.edu.cn/PLMAlign](https://dmiip.sjtu.edu.cn/PLMAlign/) :airplane:

PLMSearch  web server : [dmiip.sjtu.edu.cn/PLMSearch](https://dmiip.sjtu.edu.cn/PLMSearch/) 🚀

PLMSearch source code : [github.com/maovshao/PLMSearch](https://github.com/maovshao/PLMSearch/) :helicopter:

## Requirements
<span id="requirements"></span>

Follow the steps in [requirements.sh](requirements.sh)

## Data preparation
<span id="data-preparation"></span>

We have released our experiment data in [plmalign_data](https://dmiip.sjtu.edu.cn/PLMAlign/static/download/plmalign_data.tar.gz).
```bash
wget https://dmiip.sjtu.edu.cn/PLMAlign/static/download/plmalign_data.tar.gz
tar zxvf plmalign_data.tar.gz
```

## Run PLMAlign
<span id="pipeline"></span>

- Run PLMAlign locally by following the example in [pipeline.ipynb](pipeline.ipynb)

**Notice: the inputs and outputs of the example are saved in** `example/`.

## Reproduce all our experiments
<span id="main"></span>

Reproduce all our experiments with good visualization by following the steps in:
- Malidup: [malidup.ipynb](malidup.ipynb)
- Malisam: [malisam.ipynb](malisam.ipynb)

**Notice: Detailed results are saved in** `data/alignment_benchmark/result/`.

- SCOPe40: [scope40.ipynb](scope40.ipynb)

**Notice: Detailed results are saved in** `data/scope40_test/output/`.

## Citation
<span id="citation"></span>
Liu, W., Wang, Z., You, R. et al. PLMSearch: Protein language model powers accurate and fast sequence search for remote homology. Nat Commun 15, 2775 (2024). https://doi.org/10.1038/s41467-024-46808-5
