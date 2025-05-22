# run_pipeline.py

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
# Add other necessary imports

def main():
    # Data Ingestion
    ingestion = DataIngestion()
    train_data, test_data = ingestion.initiate_data_ingestion()

    # Data Transformation
    
    transformation = DataTransformation()
    train_transformed, test_transformed, preprocessor_obj_file_path = transformation.initiate_data_transformation(train_data, test_data)

    # Model Training
    trainer = ModelTrainer()
    trainer.initiate_model_trainer(train_transformed, test_transformed)

    # Add other stages as needed

if __name__ == "__main__":
    main()
