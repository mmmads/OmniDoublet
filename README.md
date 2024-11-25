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
import scanpy as sc
import omnidoublet as omnid
RNAadata = sc.read_h5ad('./OmniDoublet-main/test/RNA_new.h5ad')
ATACadata = sc.read_h5ad('./OmniDoublet-main/test/ATAC_new.h5ad')
modality = "ATAC"
Omni = omnid.OmniDoublet(RNAadata, ATACadata, modality)
omnid_res = Omni.core()


```

#### Input
RNAadata : RNA adata
modality_adata : The other modality adata.
modality : "ATAC" or "ADT" (default : "ATAC")


#### Return
OmniDoublet returns a pandas dataframe with two columns, column `score` is the predicted doublet score, while column `class` is the predicted label.
