import torch

# ... (all'inizio di RealTimeEstimator.__init__)

    def __init__(self, config):
        # 1. Gestione Dispositivo Flessibile
        requested_device = config['inference']['device'].lower()
        
        if requested_device == 'cuda' and torch.cuda.is_available():
            actual_device = 'cuda'
        else:
            if requested_device == 'cuda':
                print("AVVISO: CUDA richiesto ma non disponibile. Passaggio a CPU.")
            actual_device = 'cpu'
            
        self.device = actual_device
        config['inference']['device'] = self.device # Aggiorna la config con il device effettivo
        
        # 2. Caricamento Modello
        self.model_wrapper = DepthAnythingV2Model(
            model_name=config['model']['name'],
            device=self.device
        )
        
        # ... (resto dell'inizializzazione)

    # ... (nel process_frame)

    def process_frame(self, frame: np.ndarray):
        # ...
        # 2. Inferenza: Chiama il metodo del wrapper
        raw_depth_output = self.model_wrapper.inference(input_tensor)
        # ...
