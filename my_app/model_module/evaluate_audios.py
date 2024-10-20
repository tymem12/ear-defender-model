from torch.utils.data import DataLoader 
import torch
from typing import List
import os
from my_app.model_module.model_manager import ModelManager
from my_app.model_module.dataset import Dataset_Custom
from my_app.model_module.prediction_pipline.model_factory import PredictionPipeline

# def predict(model_manager: ModelManager, file_paths : List[str], base_dir = os.getenv('AUDIO_STORAGE')):
def predict(model_manager:PredictionPipeline , file_paths : List[str], base_dir = os.getenv('AUDIO_STORAGE')):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # if we need to chane the configuraton file we would do it here
    # model = model_manager.initialize_model()
    # model.eval()
    fname_list = []
    fragment_list = []
    model_output_list= []


    for file_path in file_paths:
        dataset = Dataset_Custom(list_IDs=[file_path], base_dir=base_dir)
        data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False) 
        for batch_x, (file_name, idx) in data_loader:
            
            batch_x = batch_x.to(device)
            
            batch_out = model_manager.predict(batch_x)
            # postprocessing = model_manager.postprocessing_fun
            # batch_label = postprocessing(batch_out)
            

            idx_list = [t.item() for t in idx]           # Extracting int from each tensor in the list
            # batch_label_list = [t.item() for t in batch_label]
            # batch_score_list = [t[-1].item() for t in batch_out]


            # Add outputs for current batch
            fname_list.extend(file_name)
            fragment_list.extend(idx_list)
            model_output_list.extend(batch_out)
            # score_list.extend(batch_score_list)
            # label_list.extend(batch_label_list)

            

        
    return fname_list, fragment_list,model_output_list






    
