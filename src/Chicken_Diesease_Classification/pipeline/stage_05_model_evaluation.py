from src.Chicken_Diesease_Classification.config.configuration import ConfigurationManager
from src.Chicken_Diesease_Classification.components.model_evaluation import ModelEvaluation
from src.Chicken_Diesease_Classification import logger

STAGE_NAME = "Model evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        print(model_evaluation_config)
        model_evaluation_config.log_into_mlflow()


if __name__ == '__main__':
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} started <<<<<<<")

    except Exception as e:
        logger.exception(e)
        raise e
