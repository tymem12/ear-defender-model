from torch.utils.data import DataLoader 
import torch
from typing import List
import os
from my_app.model_module.dataset import Dataset_Custom
from my_app.model_module.prediction_pipline.model_factory import PredictionPipeline

# def predict(model_manager: ModelManager, file_paths : List[str], base_dir = os.getenv('AUDIO_STORAGE')):
def predict(prediction_pipline:PredictionPipeline , file_paths : List[str], base_dir = os.getenv('AUDIO_STORAGE')):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    fname_list = []
    fragment_list = []
    model_output_list= []


    for file_path in file_paths:
        dataset = Dataset_Custom(list_IDs=[file_path], base_dir=base_dir)
        data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False) 
        for batch_x, (file_name, idx) in data_loader:
            
            batch_x = batch_x.to(device)
            
            batch_out = prediction_pipline.predict(batch_x)
            

            idx_list = [t.item() for t in idx]          
            # Add outputs for current batch
            fname_list.extend(file_name)
            fragment_list.extend(idx_list)
            model_output_list.extend(batch_out)

            

        
    return fname_list, fragment_list,model_output_list






    
