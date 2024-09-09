# ONLSM
An optimized non-landslide sampling method for Landslide susceptibility evaluation using machine learning models.
Hardware requirements: Intel(R) Core(TM) i5-10500 CPU @ 3.10GHz   3.10 GHz; 
RAM: 16.0 GB;
Program language: Python;
Software required: PyCharm Community Edition 2020.3.3 x64;

     We obtained the ground deformation rates using Interferometric Synthetic Aperture Radar (InSAR), and calculated the landslide proneness based on bias standardized information value (BSIV) model by using the landslide-causing factors (e.g., geology, topography, hydrology and environment). Then, we selected non-landslide samples from the points with ground deformation rates between +5 mm/yr and -5 mm/yr in very low susceptible level areas. "RDM_train. csv", "Buffer_train. csv", "Slope_train. csv", "InSAR_train. csv" and "BSIW_train. csv" were obtained using traditional non landslide sampling methods, respectively."ONLSM_train.csv" was obtained using the non landslide sampling method proposed in this article.
    Then,  we evaluated the landslide susceptibility in Wanzhou section of the Three Gorges Reservoir (China) using support vector machine (SVM), random forest (RF) and gradient boosting decision tree (GBDT) models. "RDM_ML.py, Buffer_ML.py, Slope_ML.py, InSAR_ML.py, BSIV_ML.py, and ONLSM_ML.py are the main functions used to calculate  landslide susceptibility evaluation based on different non-landslide sampling methods.

