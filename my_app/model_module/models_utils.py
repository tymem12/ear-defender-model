import torch
import os

def mesonet_postprocessing(batch_pred):
    batch_pred = torch.sigmoid(batch_pred)
    batch_pred_label = (batch_pred + 0.5).int()
    return batch_pred_label


def wav2vec_postprocessing(threshold: float):
    def postprocessing_model(batch_pred):
        # softmax = torch.nn.Softmax(dim=0)  # Softmax along the first dimension (for each tuple)
        elements = [ scor[1] for scor in batch_pred]
        label = [(tensor > float(threshold)).int()for tensor in elements ]
        return label
    return postprocessing_model