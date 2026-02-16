import torch
import torch.nn as nn
import os

def check_lstm_model():
    print("🔍 CHECKING LSTM MODEL")
    print("=" * 50)
    
    model_path = './ravdess_emotion_lstm_best copy.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Available files:")
        for file in os.listdir('.'):
            if file.endswith('.pth'):
                print(f"  - {file}")
        return
    
    try:
        # Try loading the model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"✅ Model file loaded successfully!")
        print(f"Model type: {type(checkpoint)}")
        
        # Check if it's a state dict
        if isinstance(checkpoint, dict):
            print(f"Model keys: {list(checkpoint.keys())}")
            
            # Print layer shapes
            print(f"\nLayer shapes:")
            for key, value in checkpoint.items():
                if hasattr(value, 'shape'):
                    print(f"  - {key}: {value.shape}")
                else:
                    print(f"  - {key}: {type(value)}")
        
        else:
            print("It's not a dictionary - might be a full model")
            if hasattr(checkpoint, 'state_dict'):
                print("Has state_dict method")
                state_dict = checkpoint.state_dict()
                for key, value in state_dict.items():
                    print(f"  - {key}: {value.shape}")
    
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_lstm_model()
