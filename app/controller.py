from uuid import UUID

from model_module.evaluate_dataset import predict


def predict_audios(analysis_id: UUID, selected_model: str, file_paths: list[str]):
    # get mo
    for link in file_paths:
        _, segments_list, labels = predict(selected_model, [link])
        # update database to load to the specific document 
