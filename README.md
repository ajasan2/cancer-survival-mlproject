# [Survival Time Estimation Model for Cancer Treatments](http://18.223.3.107/)

This project is an end-to-end implementation of a machine learning model that predicts survival time for patients with brain cancer. The project covers all stages from data ingestion and transformation to model training, prediction, and deployment. The system is deployed using AWS ECR and EC2 services, Docker, and GitHub Actions for CI/CD.

## Project Overview

- **Data Ingestion**: Collecting and preprocessing the data.
- **Data Transformation**: Feature engineering and scaling to prepare the data for model training.
- **Model Training**: Developing a predictive model using machine learning techniques.
- **Prediction Pipeline**: Creating a pipeline for making predictions based on new data.
- **Backend Development**: A Flask-based backend that provides an API for the model predictions.
- **Deployment**: Containerizing the application and deploying it using AWS services.

## Local Setup and Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/ajasan2/cancer-survival-mlproject
    cd cancer-survival-mlproject
    ```

2. **Build the Docker image**:
    ```bash
    docker build -t cancer-survival-mlproject .
    ```

3. **Run the Docker container**:
    ```bash
    docker run -p 5000:5000 cancer-survival-mlproject
    ```

4. **Access the Flask app**:
    ```bash
    http://localhost:5000
    ```