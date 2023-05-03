import torch
from torch import nn

class SpeechPrompter(nn.Module):
    """Speech prompting mechanism for duration, pitch and diffusion models.
    
    Provides speech prompts during training and inference to enable in-context 
    learning and zero-shot generalization.
    """
    def __init__(self, train_prompt_length, infer_prompt_length, 
                 diffusion_model, pitch_predictor, duration_predictor):
        super().__init__()
        self.train_prompt_length = train_prompt_length    # Length of prompt in sec during training  
        self.infer_prompt_length = infer_prompt_length    # Length of prompt in sec during inference
        self.diffusion_model = diffusion_model
        self.pitch_predictor = pitch_predictor
        self.duration_predictor = duration_predictor
        
    def forward(self, speech, durations, pitches, type='inference'):
        """Performs forward pass of prompt mechanism.
        
        Args:
            speech (torch.Tensor): Ground truth speech waveform, (B, T)
            durations (torch.Tensor): Ground truth durations, (B, L)
            pitches (torch.Tensor): Ground truth pitches, (B, L)
            type (str): 'training' or 'inference'
            
        Returns:
            latent_vectors (torch.Tensor): Predicted latent vectors, (B, L, D)
            pred_durations (torch.Tensor): Predicted durations, (B, L)
            pred_pitches (torch.Tensor): Predicted pitches, (B, L)
            loss (torch.Tensor): Loss between predictions and ground truth
        """
        B, T = speech.shape
        L = durations.shape[1]
        
        # During training, take prompt from start of speech
        if type == 'training': 
            # Get prompt length in samples and take from start of speech
            prompt_length = int(self.train_prompt_length * sampling_rate)  
            prompt = speech[:, :prompt_length] 
        
        # During inference, take prompt from input
        elif type == 'inference':
            # Get prompt length in samples and take from start of speech
            prompt_length = int(self.infer_prompt_length * sampling_rate)  
            prompt = speech[:, :prompt_length]  
            
            # The models will then predict the subsequent sequence in a zero-shot manner
            # to generate the full speech waveform  
            
        # Check if prompt is longer than speech
        if prompt.shape[1] > T: 
            raise ValueError('Prompt length exceeds speech length!')
            
        latent_vectors = []
        pred_durations = []
        pred_pitches = []
        losses = []
        
        # For each sample in the batch
        for b in range(B):  
            
            # Get prompt speech for current sample
            prompt_speech = prompt[b:b+1]  
            
            # Have models predict subsequent latent vectors and targets
            latent_pred = self.diffusion_model(prompt_speech)
            dur_pred = self.duration_predictor(prompt_speech)
            pitch_pred = self.pitch_predictor(prompt_speech)
        
            # Calculate losses between predictions and ground truth
            diffusion_loss = nn.MSELoss()(latent_pred, latent_gt[b:b+1])  
            dur_loss = nn.MSELoss()(dur_pred, durations[b:b+1])
            pitch_loss = nn.MSELoss()(pitch_pred, pitches[b:b+1])
            loss = diffusion_loss + dur_loss + pitch_loss
            
            latent_vectors.append(latent_pred)
            pred_durations.append(dur_pred)
            pred_pitches.append(pitch_pred)
            losses.append(loss)
            
            # Update prompt for next iteration
            prompt_speech = torch.cat((prompt_speech, latent_pred), dim=1)  
                             
       
        # Stack individual sample predictions
        latent_vectors = torch.cat(latent_vectors, dim=0)
        pred_durations = torch.cat(pred_durations, dim=0)
        pred_pitches = torch.cat(pred_pitches, dim=0)
        loss = torch.mean(torch.stack(losses))
        
        return latent_vectors, pred_durations, pred_pitches, loss


