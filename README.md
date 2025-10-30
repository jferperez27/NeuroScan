
# <div align="center" style="color:blue">Neuro-Scan 🧠</div>

<div align="center">
A lightweight MRI scan visualizer and deep learning classifer for AD (Alzheimer's Disease) Dementia 
and CD (Cognitively Normal) patients.
<br><br>
This is all made possible by the (Open Access Series of Imaging Studies) OASIS-3 (Longitudinal Multimodal 
Neuroimaging, Clinical, and Cognitive Dataset for Normal Aging and Alzheimer's Disease) dataset by Washington 
University in St. Louis Medicine. Full credits/reference can be found below.
</div>

# <div align="center" style="color: blue"> Roadmap </div>
- OASIS-3 dataset verification

- OASIS-3 dataset preprocessing

- MRI scan preprocessing

- TensorFlow 3D model created with small, sample dataset <---------------(in progress...)

- TensorFlow 3D model created with full dataset

- Reach at least 80% accuracy

- Tensorflow inference backend API via TypeScript

- Web app UI designed and implemented for improved user experience

- Web app supports visualization of user-uploaded MRI scans

- TypeScript backend (in addition to predictions) returns a heatmap for "areas of interest" regarding the user-uploaded MRI scan

- The web app front-end is able to visualize the heatmap returned by the backend to highlight areas of the MRI scan that resulted in the TensorFlow model classifying it as AD (alzheimer's disease) Dementia or CD (cognitively normal)

- Enable support for other common brain diseases to classify and detect
## <div align="center" style="color:blue">Credits</div>

**OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset for Normal Aging and Alzheimer Disease**
Pamela J LaMontagne, Tammie L.S. Benzinger, John C. Morris, Sarah Keefe, Russ Hornbeck, Chengjie Xiong, Elizabeth Grant, Jason Hassenstab, Krista Moulder, Andrei Vlassenko, Marcus E. Raichle, Carlos Cruchaga, Daniel Marcus, 2019. medRxiv. doi: 10.1101/2019.12.13.19014902

**About OASIS-3:**

*OASIS-3 is a retrospective compilation of data for 1378 participants that were collected across several ongoing projects through the WUSTL Knight ADRC over the course of 30years. Participants include 755 cognitively normal adults and 622 individuals at various stages of 
cognitive decline ranging in age from 42-95yrs. All participants were assigned a new random identifier and all dates were removed and normalized to reflect days from entry into study. The dataset contains 2842 MR sessions which include T1w, T2w, FLAIR, ASL, SWI, time of flight, 
resting-state BOLD, and DTI sequences. Many of the MR sessions are accompanied by volumetric segmentation files produced through FreeSurfer processing. PET imaging from different tracers, PIB, AV45, and FDG, totaling over 2157 raw imaging scans and the accompanying post-processed 
files from the Pet Unified Pipeline (PUP) are also available in OASIS-3. Additionally, 451 Tau PET sessions and post-processed PUP are now available for OASIS-3 subjects in a sub-project ‘OASIS-3_AV1451’.*

[OASIS-3 Website](https://sites.wustl.edu/oasisbrains/home/oasis-3/)

## <div align="center" style="color:blue">Contributing </div>
We're looking to expand our team!  

Email me at juan_j_fernandez@brown.edu with your information and we can begin the process of adding you to our team.
We have some who specialize in solely backend, frontend, machine learning, or none at all. Regardless of your expertise, we encourage you to reach out!

**Note: You <u>MUST</u> be able to access the OASIS-3 dataset. Thus, you will have to start the process of verifying yourself as accessible user to the OASIS-3 dataset and get approved in order to work in the project.**
