from Consumer_Complaint_Analysis.config import ConfigurationManager
from Consumer_Complaint_Analysis.components import PrepareBaseModel, DataPreProcessing
from Consumer_Complaint_Analysis import logger

STAGE_NAME = "Prepare base model"

def main():
    config = ConfigurationManager()
    prepare_base_model_config = config.get_prepare_base_model_config()
    Data = DataPreProcessing(config=prepare_base_model_config)
    df = Data.pre_process_data(prepare_base_model_config.file_path)
    prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
    prepare_base_model.get_base_model(df)
if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e