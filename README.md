# OmniDoublet
OmniDoublet : a doublet detection method for multimodal single-cell data

### Installation
```
git clone https://github.com/mmmads/OmniDoublet
cd OmniDoublet
pip install -r requirements.txt 
python setup.py install
```

### Quick Start
```

```

#### Input
RNAadata : RNA adata
modality_adata : The other modality adata.
modality : "ATAC" or "ADT" (default : "ATAC")


#### Output
OmniDoublet output a file `omnid_res.csv` with two columns, column `score` is the predicted doublet score, while column `class` is the predicted label.