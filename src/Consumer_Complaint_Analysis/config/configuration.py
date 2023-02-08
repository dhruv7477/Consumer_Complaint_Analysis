from Consumer_Complaint_Analysis.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from Consumer_Complaint_Analysis.utils import read_yaml, create_directories
from pathlib import Path
import os
from Consumer_Complaint_Analysis.entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig
)

class ConfigurationManager:
    def __init__(
        self, 
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config
        params = self.params
        create_directories([config.prepare_base_model.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.prepare_base_model.root_dir),
            base_model_path=Path(config.prepare_base_model.base_model_path),
            file_path=Path(config.data_ingestion.csv_file_path),
            params_num_labels=params.NUM_LABELS,
            params_test_size=params.TEST_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_random_state=params.RANDOM_STATE,
            params_batch_size=params.BATCH_SIZE
        )

        return prepare_base_model_config