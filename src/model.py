import torch
from depth_anything_v2.dpt import DPT_DepthModel

class DepthAnythingV2Model:
    """
    Wrapper per caricare e gestire l'inferenza di Depth Anything V2.
    """
    
    def __init__(self, model_name: str, device: str = 'cpu'):
        """
        Inizializza il modello DPT.
        :param model_name: Esempio: 'metric_depth_vit_large'
        :param device: 'cuda' o 'cpu'
        """
        self.device = torch.device(device)
        self.model = self._load_model(model_name)
        self.model.eval().to(self.device)
        print(f"Depth Anything V2 Model '{model_name}' caricato su {self.device}.")

    def _load_model(self, model_name: str):
        """ Carica il modello pre-addestrato dalla libreria. """
        
        # Le librerie SOTA spesso scaricano i pesi automaticamente.
        model = DPT_DepthModel(
            name=model_name,
            backbone='vitl',  # Usa il backbone corretto per il tuo modello (e.g., vitl, vits, vitg)
            non_negative=True  # Assicura output non negativi
        )
        # La funzione 'load_state_dict' gestisce il download dei pesi corretti
        # se non li trova in locale.
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            model.get_checkpoint_url(model_name),
            map_location=self.device
        ))
        return model

    def inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """ Esegue l'inferenza su un tensore pre-processato. """
        with torch.no_grad():
            # Sposta l'input sul dispositivo corretto
            input_tensor = input_tensor.to(self.device)
            # L'output è la mappa di profondità grezza/disparità
            depth_output = self.model(input_tensor)
        return depth_output
