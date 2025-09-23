# HeartAttackLikeliness

A machine learning project for predicting heart attack likelihood based on various health indicators and lifestyle factors.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a machine learning model to predict the likelihood of heart attacks based on various health and lifestyle parameters. The model analyzes factors such as age, blood pressure, cholesterol levels, and other relevant medical indicators to provide risk assessments.

## Features

- **Data Analysis**: Comprehensive exploratory data analysis of heart disease risk factors
- **Machine Learning Models**: Implementation of multiple ML algorithms for comparison
- **Risk Prediction**: Predict heart attack likelihood for new patient data
- **Visualization**: Interactive charts and graphs showing data insights
- **Model Evaluation**: Detailed performance metrics and validation

## Requirements

- Python 3.7 or higher
- Dependencies listed in `requirements.txt`

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Akasi132/HeartAttackLikeliness.git
cd HeartAttackLikeliness
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Running the Analysis

1. **Ensure your virtual environment is activated**:
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Run the main analysis script**:
   ```bash
   python HeartAttackAnalysis.py
   ```

### Input Data Format

The model expects the following features:
- Age
- Sex (0 = female, 1 = male)
- Chest pain type (0-3)
- Resting blood pressure
- Serum cholesterol level
- Fasting blood sugar (0 = false, 1 = true)
- Resting electrocardiographic results (0-2)
- Maximum heart rate achieved
- Exercise induced angina (0 = no, 1 = yes)
- ST depression induced by exercise
- Slope of peak exercise ST segment (0-2)
- Number of major vessels colored by fluoroscopy (0-3)
- Thalassemia type (0-3)

## Dataset

The project uses the `heart_attack_prediction_dataset.csv` file containing anonymized patient data with the following characteristics:
- **Size**: Multiple patient records with 13+ features
- **Target Variable**: Heart attack occurrence (0 = no, 1 = yes)
- **Features**: Mix of categorical and continuous variables

## Model Performance

The model performance metrics will be displayed when running the analysis, including:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

## Troubleshooting

### Common Issues

#### 1. Module Import Errors
**Problem**: `ModuleNotFoundError` when running the script

**Solution**:
```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

#### 2. Python Version Compatibility
**Problem**: Script fails due to Python version

**Solution**:
- Ensure you're using Python 3.7+
- Check version: `python --version`
- Consider using `python3` instead of `python` on some systems

#### 3. Dataset Loading Issues
**Problem**: CSV file not found or corrupted

**Solution**:
```bash
# Verify file exists
ls -la heart_attack_prediction_dataset.csv

# If missing, ensure you've cloned the complete repository
git pull origin main
```

#### 4. Memory Issues
**Problem**: Script runs out of memory during processing

**Solution**:
- Close other applications to free up RAM
- Consider processing data in smaller chunks
- Upgrade system memory if persistently problematic

#### 5. Plotting/Visualization Errors
**Problem**: Matplotlib or plotting libraries not working

**Solution**:
```bash
# For headless systems, set backend
export MPLBACKEND=Agg

# Or install GUI backend
sudo apt-get install python3-tk  # Linux
```

#### 6. Virtual Environment Issues
**Problem**: Virtual environment not working properly

**Solution**:
```bash
# Remove and recreate virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### Getting Help

If you encounter issues not covered above:
1. Check the [Issues](../../issues) page for similar problems
2. Create a new issue with:
   - Your operating system
   - Python version
   - Complete error message
   - Steps to reproduce

### Performance Tips

- **First Run**: The initial run may take longer due to model training
- **Data Size**: Larger datasets will require more processing time
- **Hardware**: Multi-core processors will improve performance for certain operations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

**Note**: This model is for educational and research purposes only. Always consult with qualified healthcare professionals for medical advice and diagnosis.
