## Introduction
The goal of this project is to improve diagnostic accuracy by leveraging radiomics and imaging biomarkers extracted from medical images CT scans. The project will also explore the use of deep learning and traditional machine learning models to classify lymph node status and identify imaging biomarkers associated with patient outcomes.

## INFO
- if wavelet = True then n = processes <= 2

- for all ~classes/Types
        :::: single binWidth 1599-4 = 1595 (1363-4= 1359 when rem_dup True)    
        ;;;; two- binWidths 2594-4 = 2590 ~~ 1000 feats per binWidth
        ;;;; two- binWidths with duplicates 3194-4 = 3190 - for comparison of features reproducibility
        

        try binWidths = [5, 10, 25, 50]


## TODO

Feature selection / Dimension reduction
        - Step 1: Exclusion of nonreproducible features
        - Step 2: Selection of the most relevant variables for the respective task
        - Step 3: Building correlation clusters
        - Step 4: Data visualization - once the data dim has been reduced.
        - Step 5: Selection of most representative features for each cluster
        - Step 6: Model fitting with remaining features (usually 3-10 fts)

- radiomic features for different bin-widths (5-50?)
- LASSO 
- use range resegmentation?


## Dataset
TBC

## Methodology
TBC

## License
TBC




