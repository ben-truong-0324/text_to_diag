# text_to_diag
# Text2Diag

This repository is an implementation attempt of the research paper **"FarSight: Long-Term Disease Prediction Using Unstructured Clinical Nursing Notes"** by Khai Truong and Nicholas Grub, Georgia Tech, as part of CS7641: Machine Learning. The goal is to develop a model capable of generating diagnostic codes from unstructured nursing notes, enabling accurate diagnostic code assignment at a patient's first ICU visit.

## Overview

Clinical nursing notes contain rich, unstructured information about a patient's condition and care. Extracting actionable insights from these notes is critical for timely and accurate disease prediction and diagnosis. This project aims to replicate the methodology and findings of FarSight by employing machine learning techniques to process unstructured text and predict diagnostic codes.

## Features

- **Text Preprocessing**: Transform unstructured nursing notes into a format suitable for machine learning.
- **Model Training**: Train machine learning models to predict diagnostic codes based on the notes.
- **Evaluation**: Assess model performance using appropriate metrics, including accuracy, precision, recall, and F1-score.
- **Replication of FarSight**: Closely replicate the steps and findings outlined in the research paper.

## Getting Started

### Prerequisites

Set up the environment using the `bd4h` configuration provided:

#### Environment Setup

1. Install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Create the environment using the provided configuration file:

   ```bash
   conda env create -f environment.yml
