# FedMelanom: Medical Support Platform for the Real-Time Detection and Analysis of Melanomas

## Overview

**FedMelanom** is an intelligent medical platform designed to assist healthcare professionals in the early detection of melanomas, a severe type of skin cancer. The solution integrates Deep Learning and Federated Learning techniques to analyze dermatological images while preserving patient data privacy.  
It includes an API and an intuitive web application for clinical interaction.
---

## Repository Structure

This repository is organized as follows:

- **api/**: Contains the FastAPI backend, which exposes the main endpoints for model training, evaluation, retraining, federated learning, and data management. In addition to these core functions, the API also includes a set of auxiliary utilities to support data preprocessing, model versioning, evaluation metrics, and federated coordination.
- **front/**: Includes the Angular-based web application that enables clinicians to interact with the system, upload images, visualize results, and generate clinical reports.


## Prerequisites

To set up and run FedMelanom locally, ensure that your environment meets the following requirements:

- **Python 3.10**
- **Node.js** (version 18 or later) and **npm**
- **pip** (Python package manager)
- (Optional) **GPU support** (CUDA/cuDNN) for local model training

## API

The `api/` directory contains the FastAPI backend, which implements all core and auxiliary functionalities such as model training, evaluation, retraining, federated learning coordination, and comprehensive data management.

### Environment Requirements

- Python 3.10.x
- Windows operating system
- NVIDIA GPU with CUDA 11.8 support (recommended for accelerated model training)

> **Note:**  
> The API is designed to take advantage of an NVIDIA GPU with CUDA 11.8 and cuDNN for optimal performance during training and retraining.  
> If a compatible GPU is not available, the system will automatically use the CPU, but training will be considerably slower.

### Required Python Packages

The main dependencies are:

- fastapi==0.110.0
- uvicorn==0.34.2
- python-multipart==0.0.20
- numpy==1.26.4
- pandas>=1.5.3
- matplotlib==3.8.3
- opencv-python==4.9.0.80
- scikit-learn==1.6.1
- tensorflow==2.10.1
- keras==2.10.0
- modelaverage==1.0.1

You can install all dependencies by running:

```bash
pip install -r requirements.txt
   ```

### Running the API

Follow these steps to set up the environment, install dependencies, and run the FastAPI server:

1. **Create the virtual environment:**

    ```bash
    python -m venv fastapi-env
    ```

2. **Activate the virtual environment:**

    ```bash
    fastapi-env\Scripts\activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Deploy the API:**  
    Make sure the virtual environment is activated before running the server.

    To launch the API using Uvicorn, use one of the following commands:

    ```bash
    uvicorn main:app
    ```
    or
    ```bash
    python -m uvicorn main:app
    ```

    The API will be available by default at [http://localhost:8000](http://localhost:8000).

> **Tip:**  
> You can access the automatic API documentation provided by FastAPI at [http://localhost:8000/docs](http://localhost:8000/docs).

---

#### Running Auxiliary Functions

In addition to serving as an API, the main script also contains several auxiliary functions for tasks such as base model training, evaluation, etc. 
**To use these utilities, it is not necessary to launch the FastAPI server.**  
You can create a separate `main` block within the same `.py` file and execute it directly with the required function calls.

For example:

```python
if __name__ == "__main__":
    # Example of calling a training function
    train_base_model()
```

## API Directory Structure
> **Note:** Some directories described below (such as `data/`, `images/`, `models/`, and `checkpoints/`) are not present in the repository due to storage constraints, but are documented here for reference and local use.


The `api/` directory is organized as follows:

- **checkpoints/**  
  Stores model checkpoints saved after each retraining cycle. These are used to resume models that have already been retrained, rather than using the base or federated models.

- **data/**  
  This folder is intended to contain all datasets used for training and evaluation.  
  **However, due to their large size, images and dataset files are not included in the repository.**  
    - **isic2018/**: The full ISIC 2018 Challenge dataset, with images and labels used for initial training and validation.
    - **isic2018Fed/**: Contains partitioned subsets of the ISIC2018 dataset, simulating three independent medical centers. Images and labels are split via a custom script (`redistribucion.py`), with each subfolder (e.g., `medico1`, `medico2`, `medico3`) representing a different center.
- **evaluations/**  
  (If used) Stores results or logs from evaluation runs.

- **images/**  
  This directory is subdivided into folders named after both patient IDs and user (medical center) IDs.  
  When clinicians upload images via the web application for prediction, each image is stored under:
  - The **corresponding user (medical center) subfolder**, which contains all images uploaded by that center’s users and is used for retraining the models for each center.
  - The **patient subfolder**, which organizes images per individual patient for traceability and future reference.

  This dual structure allows accurate patient-level tracking and enables efficient per-center retraining workflows.  
  The `images/` directory is always used in conjunction with the `labels/` directory to associate predictions and labels with the respective images.

- **json/**  
  Contains various JSON files for platform configuration and metadata.
    - **informs/**: Generated clinical reports in JSON format for each medical center.
    - **patients.json**, **users.json**: Store information about patients and users (medical centers), respectively.

- **labels/**  
  Stores the label files generated after model predictions. Each subfolder (named after a medical center's ID) contains the corresponding prediction labels attached to images, supporting retraining and federated aggregation.

- **models/**  
  This folder is intended to store all trained model files.  
  **However, due to their large size, model files are not included in the repository.**
    - **basic/**: Contains a centralized model trained on a large (10,000-image) dataset for comparison against federated models.
    - **fed/**: Stores the base model for federated learning, as well as retrained models for each center.
    - **average/**: Contains the federated averaged models.

- **fastapi-env/**  
  (Local only) Python virtual environment directory (should be excluded from the repository).


- **main.py**  
  Main script for launching the FastAPI server and exposing API endpoints. Also contains auxiliary functions for training, evaluation, and federated learning that can be executed independently without starting the server.

- **redistribucion.py**  
  Script for partitioning the original ISIC2018 dataset to simulate multiple medical centers, creating separate folders and label files for each.



---

### **Notes**
- Large datasets and trained model files are not included in the repository.  
  Please follow the dataset preparation instructions to generate or download these files as needed.
- The provided scripts ensure consistent dataset partitioning and model management across all centers in federated learning scenarios.



## Web Application

The `front/` directory contains the Angular-based frontend of FedMelanom. This application provides clinicians with an intuitive interface to upload images, visualize predictions, access model evaluation results, and download clinical reports.

### Prerequisites

- Node.js (version 18 or later)
- npm
- Angular CLI (version 17.0.10 or later)

### Setup and Development

1. **Install dependencies:**

    ```bash
    npm install
    ```

2. **Start the development server:**

    ```bash
    ng serve
    ```

    The application will be available at [http://localhost:4200](http://localhost:4200). The app supports live reload on file changes.

### Building the Project

To create a production-ready build, run:

```bash
ng build
```

# .gitignore recommendations
It is recommended to exclude the following from version control:
- fastapi-env/
- data/
- images/
- labels/
- models/
- checkpoints/