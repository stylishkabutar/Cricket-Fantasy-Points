# Cricket-Fantasy-Points
This is a system to predict the fantasy points made by a player in a given match. The current apparatus is for IPL.
# To Generate Data for other formats
Use the code in the Final Datagen file to generate Data files. The inputs are .YAML files(preferably from cricsheet- as we have adjusted for their naming conventions).
# Models
To train models for batters, fielders and bowlers- 3 notebooks are provided. All of them have a Keras tuner module which will give the most optimal model architecture.
We have use a new Embeddings Based Architecture- this provides SOTA performance for Bowlers and fielders. For batters we still rely on XGBoost.
