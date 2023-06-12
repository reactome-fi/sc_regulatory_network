## Gene Expression Regulatory Network Construction with TFs and Pathways using scRNA-seq Data

### Description:
This repository provides the necessary code to construct gene expression regulatory networks utilizing single-cell RNA sequencing (scRNA-seq) data. These networks incorporate both transcription factors (TFs) and pathways obtained from Reactome (https://reactome.org). The code relies on the scpy4reactome package, which can be installed by following the instructions available at https://github.com/reactome-fi/fi_sc_analysis/tree/master/python.

### Repository Contents:
- Python script to build regulatory networks using scRNA-seq
- Example notebook demonstrating usage

### Installation:
To utilize this codebase, you will need to install the scpy4reactome package. Follow the instructions provided at https://github.com/reactome-fi/fi_sc_analysis/tree/master/python for installation details.

### Usage:
1. Preprocess your scRNA-seq data to prepare it for network construction.
2. Ensure that the scpy4reactome package is correctly installed.
3. Run the provided code script or notebook to construct gene expression regulatory networks.
4. Utilize the resulting networks to analyze TF and pathway interactions and their impact on gene expression regulation.
5. Load the network into ReactomeFIViz, a Cytoscape App, for fuzzy logic based simulation. For more information about this, see https://reactome.org/userguide/reactome-fiviz. 

Note: Additional instructions and guidelines will be provided within the repository soon to assist with specific usage scenarios.

### Contributing:
Contributions, bug reports, and feature requests are welcome.

### License:
The code in this repository is available under the Apache License Version 2.0. Please review the LICENSE file for detailed information.

### Contact:
For any inquiries or questions regarding this project, please reach out to us via GitHub Issues.
