from torch.utils.data import DataLoader 
import torch
from model_module.model_manager import ModelManager
from model_module.test_dataset import Dataset_Custom

def predict(model_name: str, file_paths : list[str]):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_manager = ModelManager(model_name)
    # if we need to chane the configuraton file we would do it here
    model = model_manager.initialize_model()
    model.eval()
    fname_list = []
    fragment_list = []
    score_list = []
    label_list = [] 


    for file_path in file_paths:
        dataset = Dataset_Custom(list_IDs=file_path)
        data_loader = DataLoader(dataset, batch_size=14, shuffle=False, drop_last=False) 
        for batch_x, (file_name, idx) in data_loader:
            
            
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)
            
            batch_out = model(batch_x)
            postprocessing = model_manager.postprocessing_fun
            batch_label = postprocessing(batch_out)
            
            batch_score = batch_out.data.cpu().numpy()
            # Add outputs for current batch
            fname_list.extend(file_name)
            fragment_list.extend(idx)
            score_list.extend(batch_score)
            label_list.extend(batch_label)
        
    return fname_list, fragment_list, label_list






    # TU SIĘ ZACZYNA TERAZ PRACA: 
    # NIE TWORZYMY DATASETU Z CAŁEGO ZBIORU ALE PLIK BY PLUK

    