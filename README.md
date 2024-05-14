# Car Evaluation Predictor

## Overview
This project is an end-to-end machine learning pipeline that predicts car evaluations based on various attributes like buying price, maintenance price, number of doors, person capacity, size of luggage, and safety levels. The machine learning model has been trained using the Car Evaluation Data Set from the UCI Machine Learning Repository. The model and its preprocessor are deployed as a FastAPI endpoint hosted on AWS, providing a robust and scalable API for real-time predictions.

## Features
- **Data Preprocessing**: Implements a sophisticated data preprocessing pipeline that handles categorical and ordinal features, ensuring optimal model performance.
- **Model Training**: Utilizes several machine learning algorithms, with a final model chosen based on performance metrics such as accuracy and precision.
- **API Development**: FastAPI is used to create a high-performance, easy-to-use RESTful API for making predictions.
- **Cloud Deployment**: The model and API are containerized using Docker and deployed on AWS, leveraging AWS EC2 for scalable and efficient application hosting.

## Technologies Used
- **Python**: Primary programming language for development.
- **Pandas** & **NumPy**: For data manipulation and numerical operations.
- **Scikit-Learn**: For building machine learning models.
- **Joblib**: For model serialization and deserialization.
- **FastAPI**: For the web framework to serve the API.
- **Uvicorn**: As the ASGI server for FastAPI.
- **Docker**: For containerizing the application.
- **AWS EC2**: For hosting the Docker container and deploying the API.
- **Git**: For version control.

## Getting Started

### Prerequisites
- Python 3.8+
- Docker
- AWS CLI configured with administrative access to your AWS account

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/taaha3244/mlops_cars.git
   

2. **Install the dependencies**
    ```bash
   pip install -r requirements.txt

3. **Start the FastAPI server**
   ```bash
   uvicorn app:app --reload

### Contributing
Contributions are welcome! Please fork the repository and create a pull request with your features or fixes.

### Contact
Taaha Mushtaq – taaha.com@gmail.com




