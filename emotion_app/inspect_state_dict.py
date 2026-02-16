# inspect_state_dict.py
import torch
import os
from collections import OrderedDict

MODEL_PATH = os.path.join("Emotion_From_Speech_Model2", "pytorch_model.bin")

def main():
    print("Loading:", MODEL_PATH)
    sd = torch.load(MODEL_PATH, map_location="cpu")
    # If a checkpoint dict (with optimizer etc), try to find 'state_dict'
    if isinstance(sd, dict) and ('state_dict' in sd or 'model_state_dict' in sd):
        if 'state_dict' in sd:
            sd = sd['state_dict']
        else:
            sd = sd['model_state_dict']

    if not isinstance(sd, dict):
        print("Loaded object is not a dict. Type:", type(sd))
        return

    keys = list(sd.keys())
    print(f"NUM KEYS: {len(keys)}")
    print("First 40 keys (or fewer):")
    for k in keys[:40]:
        print("  ", k, " -> ", tuple(sd[k].shape) if hasattr(sd[k], 'shape') else type(sd[k]))

    # show some stats
    num_params = 0
    for k,v in sd.items():
        try:
            num_params += v.numel()
        except Exception:
            pass
    print("Total parameters (approx):", num_params)

    # detect common prefixes like 'module.' from DataParallel
    prefixes = set(k.split('.')[0] for k in keys if isinstance(k, str))
    print("Top-level prefixes (sample):", list(prefixes)[:10])

    # check for HF-style components
    hf_indicators = ['bert', 'wav2vec', 'transformer', 'classifier', 'lm_head', 'feature_extractor']
    present = [t for t in hf_indicators if any(t in k.lower() for k in keys)]
    print("Possible model-family indicators found in keys:", present)

    # optionally dump a trimmed JSON-like map to file for pasting
    try:
        import json
        trimmed = [{ "key": k, "shape": tuple(sd[k].shape) if hasattr(sd[k], 'shape') else None } for k in keys[:200]]
        with open("state_dict_keys_sample.json", "w") as f:
            json.dump(trimmed, f, indent=2)
        print("Wrote state_dict_keys_sample.json (first 200 keys+shapes).")
    except Exception:
        pass

if __name__ == "__main__":
    main()
