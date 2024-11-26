import logging
import os
from typing import List
import torch
from torch.utils.data import DataLoader 
from my_app.model_module.dataset import Dataset_Custom
from my_app.model_module.prediction_pipeline.model_factory import PredictionPipeline

def predict(prediction_pipeline:PredictionPipeline , file_paths : List[str], base_dir : str = None) :
    if not base_dir:
        base_dir = os.getenv('AUDIO_STORAGE')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    fname_list = []
    fragment_list = []
    model_output_list= []
    i = 0

    for file_path in file_paths:
        try:
            dataset = Dataset_Custom(list_IDs=[file_path], base_dir=base_dir)
        except FileExistsError as e:
            logging.error('Error in predict. File skipped: ' + str(e))
            continue
        data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False) 
        for batch_x, (file_name, idx) in data_loader:
            
            batch_x = batch_x.to(device)
            batch_out = prediction_pipeline.predict(batch_x)
            i +=1
            idx_list = [t.item() for t in idx]          

            # Add outputs for current batch
            fname_list.extend(file_name)
            fragment_list.extend(idx_list)
            model_output_list.extend(batch_out)
        dataset.clean_dataset()
    return fname_list, fragment_list,model_output_list






    
