#### *This is a reproducibility study of the following paper:*

# ✨GLORY: Global Graph-Enhanced Personalized News Recommendations

Code for the author's paper: [_Going Beyond Local: Global Graph-Enhanced Personalized News Recommendations_](https://arxiv.org/pdf/2307.06576.pdf) published at RecSys 2023. 

<p align="center">
  <img src="glory.jpg" alt="Glory Model Illustration" width="600" />
  <br>
  Glory Model Illustration
</p>


### Environment


```shell
cd GLORY
# To install the working environment for our code:
conda env create -f environment.yml
conda activate GLORY
```

```shell
# To download the original MINDsmall and MINDlarge dataset:
bash scripts/data_download.sh

# Run and change parameters accordingly
python3 src/main.py model=GLORY dataset=EBNeRDsmall reprocess=True model.use_entity=True model.his_size=50
```


### Entity Embeddings

To generate the entity embeddings, you must [install GraphVite](https://graphvite.io/docs/latest/install.html):
```shell
conda install -c milagraph -c conda-forge graphvite cudatoolkit=$(nvcc -V | grep -Po "(?<=V)\d+.\d+")
```

And [download the pretrained weights](https://graphvite.io/docs/latest/pretrained_model):
```shell
cd entity_emb
wget https://udemontreal-my.sharepoint.com/:u:/g/personal/zhaocheng_zhu_umontreal_ca/EVcJpJAzkThPu1vjgJLohscBgwtPajhTZvCCd8nEg1GiwA?download=1
```

In the correct conda environment with GraphVite, run:
```shell
# Get a list of all entities present in the EBNeRD-small dataset and collect them to a json file
python3 preprocess.py

# Use graphVite to query the WikiData ID and embedding vector for each entity in the json file and add it to the same json file
python3 embed.py

# Write the contents in the json file to the data folder in a txt format that GLORY can use
python3 refine.py
```

### Files that were changed or added by us:

In general the code was cleaned up too to remove unecessary commented out code by the authors.
Here is a list of all the files that were changed in the code or added. Additional precisions as to how each file was changed can be found in comments throughout the code of these different files:

- <code> configs/path/default.yaml</code>
- <code> configs/ebnerd.yaml</code> (new file created from copying small.yaml and changing some parameters)
- <code> configs/default.yaml</code>
- <code> configs/EBNeRDsmall.yaml</code> (new file created from the Ekstra Bladet dataset)
- <code> configs/GLORY_EBNeRD.yaml</code> (new file created to accomodate for the entity embeddings and change of history size)
- <code> src/main.py</code>
- <code> src/dataload/data_preprocess_parquet.py</code> (new file created from copying data_preprocess.py and changing some functions in it to accomodate the different dataset)
- <code> entity_emb/*</code> (code to generate the entity embeddings of EBNeRD)

### Bibliography

```shell
@misc{yang2023going,
      title={Going Beyond Local: Global Graph-Enhanced Personalized News Recommendations}, 
      author={Boming Yang and Dairui Liu and Toyotaro Suzumura and Ruihai Dong and Irene Li},
      year={2023},
      publisher ={RecSys},
}
```
