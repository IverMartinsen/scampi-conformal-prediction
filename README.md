# Detecting and analysing microfossils from wellbore NO 6407/6-5

This repo contains code for detecting and analysing microfossil crops in whole slide images (WSI). 
The code is intended for data produced by the Norwegian Offshore Directorate (NOD).
The project is structured as follows:
- the ***preprocessing*** directory contains code for
  - detecting and extracting crops from WSIs
  - creating latent space embeddings for extracted crops using a pretrained vision transformer (ViT)
- the ***postprocessing*** directory contains code for
  - outlier detection of crops
  - classification of crops
  - crop detection using conformal prediction
  - estimation of genus distribution across depths
