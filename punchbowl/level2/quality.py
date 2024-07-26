
from ndcube import NDCube
from prefect import get_run_logger, task


@task
def quality_flag_task(data_list: list[NDCube]) -> list[NDCube]:
    """Prefect task to perform quality flagging."""
    logger = get_run_logger()
    logger.info("quality_flag started")
    # TODO: actually do the quality flagging
    logger.info("quality_flag ended")
    for data_object in data_list:
        data_object.meta.history.add_now("LEVEL2-quality_flag", "quality flagging completed")
    return data_list
