# C1PK-Flood: A hybrid model integrating the C1PKNet and the MatFlood models
> The C1PK-Flood model is a hybrid system integrating a machine learning model (C1PKNet; Lee et al., 2021) and a static flood model (MatFlood; Enriquez et al., 2023) to generate high-resolution peak storm surge inundation maps. This model predicts the inundation area in the Galveston area based on the time series of six storm parameters.


# C1PK-Flood model running process

### C1PKNet_training.py  
> Code for training the C1PKNet model.
- The input data consists of a time series of six storm parameters, and the output data represents the peak storm surge values at 1,217 points. To determine the optimal time series, we compared a total of 215 combinations. We selected data spanning from 4 hours before landfall to 6 hours after landfall for the input time series data.

### Cross-validation_plotting.py
> Code for cross-validation result.
- Nested 10-fold cross-validation was performed, and a density scatter plot was generated to compare the simulated peak storm surge values with the C1PKNet-predicted peak storm surge values.

### Preprocessing.py
> Code for preprocessing.
- Using the original mesh information from the physics-based model, generate high-resolution DEM information utilizing the k-nearest neighbor algorithm. The DEM data is transformed to ensure compatibility with the C1PK-Flood model by converting the coordinate reference system to decimal degrees and replacing cells with NaN values with valid data. The processed file is then saved for running the C1PK-Flood model.

### main_C1PK_Flood.py
> Code for running the C1PK-Flood model.
- The results of the C1PKNet model are used as input data for the MatFlood model, which ultimately generates a high-resolution inundation map for the Galveston area in TIFF format. One of the test storms and the reduction factor for the MatFlood model can be defined. The result is output under the name RESULT_C1PK_Flood_storm, appended with the corresponding test storm number.


# Necessary Inputs Files

- **C1PK_Flood_Modules.py**: Modules for the C1PK-Flood model.  
- **config.py**: Parameters for the C1PK-Flood model.  
- **ADCIRC_ori_mesh.pkl**: The unstructured triangular computational mesh of the numerical models in the Galveston area.  
- **TRAINING_DATA.pkl**: Training dataset of the C1PKNet model.  
- **TEST_DATA.pkl**: Test dataset of the C1PKNet model. 



# Release History
* 1.0.1
    * The first proper release

# Meta

Hyunje Yang – hyunjeyang@utexas.edu

Distributed under the Creative Commons Legal Code CC0 1.0 Universal. See ``LICENSE`` for more information.

# Reference
Lee, J.W., Irish, J.L., Bensi, M.T. and Marcy, D.C., 2021. Rapid prediction of peak storm surge from tropical cyclone track time series using machine learning. Coastal Engineering, 170, p.104024. doi:https://doi.org/10.1016/j.coastaleng.2021.104024

Enriquez, A.R., Wahl, T., Talke, S.A., Orton, P.M., Booth, J.F., Agulles, M. and Santamaria-Aguilar, S., 2023. MatFlood: an efficient algorithm for mapping flood extent and depth. Environmental Modelling & Software, 169, p.105829. doi:https://doi.org/10.1016/j.envsoft.2023.105829
