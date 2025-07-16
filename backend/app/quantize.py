import torch
import torch.nn as nn
from model import CNN  
torch.backends.quantized.engine = 'qnnpack' 

model = CNN()


state_dict = torch.load('best_model.pth', map_location='cpu')
model.load_state_dict(state_dict)


model.eval()


quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

torch.jit.save(torch.jit.script(quantized_model), 'quantized_model.pt')

print("Model quantized and saved as quantized_model.pt")
