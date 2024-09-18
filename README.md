# :hugs: TransformerTraining

Welcome! 
This project focuses on training :hugs: transformers using custom datasets for classification and segmentation, in my case particularly focusing on data from Minecraft environments. 
The repository includes code for handling and processing such data, as well as scripts for model training and evaluation.

## Table of Contents
- [Warning](#warning)
- [Installation](#installation)
- [Arguments](#arguments)
- [Contributing](#contributing)
- [License](#license)

## Warning

This project is currently under development and should be considered a work in progress. The code provided here may not be fully functional or optimized for production use, and the documentation is incomplete. Use at your own risk.

## Installation

Please ensure you have Python installed on your system.

1. Clone the repository: git clone     

2. Navigate to the project directory.

3. Create a virtual environment (optional but recommended).

4. Start the virtual environment: source venv/bin/activate (on Windows use `venv\Scripts\activate`).

5. Install dependencies: pip install -r requirements.txt

6. Download or prepare your dataset and place it in the appropriate directory within the project structure.

7. Run main.py with aguments to start the script: python main.py --arg 

## Arguments

--prepareData , script is used for preparing data from Minecraft environments, converting it into a format suitable for training :hugs: transformers. This includes steps like cleaning, tokenization, and other preprocessing tasks.

--verifyData , script verifies the prepared data to ensure that it meets the required quality standards before proceeding with model training.

--train , script is used for training :hugs: transformer models using the processed data. It includes options for configuring various aspects of the training process, such as hyperparameters and optimization algorithms.

--evaluateModel , script evaluates the trained model's performance on a validation or test dataset to assess its effectiveness in handling specific tasks related to Minecraft environments.

## Contributing

Contributions are welcome! If you think something is missing, incorrect, or could be improved upon, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the "I DONT CARE License". Feel free to use it as you wish!
