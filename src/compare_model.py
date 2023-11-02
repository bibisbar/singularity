import torch







def compare_models_Qa(model_path1, model_path2, threshold=1e-5):
    # Load the model checkpoints
    model1 = torch.load(model_path1, map_location='cpu')
    model2 = torch.load(model_path2, map_location='cpu')
    
    model1_param = model1["model"]
    state_dict = model2["model"]
    has_decoder = False
    for key in list(state_dict.keys()):
        if "bert" in key:
            encoder_key = key.replace("bert.", "")
            state_dict[encoder_key] = state_dict[key]
            if not has_decoder:
                del state_dict[key]

        # init text decoder as multimodal encoder (last 6 layers of model.text_encoder)
        # only for generation tasks like VQA
        if has_decoder and "text_encoder" in key:
            if "layer" in key:
                encoder_keys = key.split(".")
                layer_num = int(encoder_keys[3])
                if layer_num < 9:  # configs/config_bert.fusion_layer
                    del state_dict[key]
                    continue
                else:
                    decoder_layer_num = (layer_num-9)
                    encoder_keys[3] = str(decoder_layer_num)
                    encoder_key = ".".join(encoder_keys)
            else:
                encoder_key = key
            decoder_key = encoder_key.replace("text_encoder", "text_decoder")
            state_dict[decoder_key] = state_dict[key]
            del state_dict[key]
    

    # Compare the model parameters
    model_diff = {}
    total_params = 0
    mismatched_params = 0

    for (name1, param1) in model1_param.items():
        if name1 not in state_dict:
            print(f"Parameter name mismatch: {name1} not in model2")
            continue
        else:
            total_params += param1.numel()
            #secondly compare the values
            param1 = param1.float()
            param2 = state_dict[name1].float()
            diff = torch.norm(param1 - param2).item()
            if diff > threshold:
                mismatched_params += 1
                model_diff[name1] = diff
    for (name2, param2) in state_dict.items():
        if name2 not in model1_param:
            print(f"Parameter name mismatch: {name2} not in model1")
            continue
        
            
    print(f"Total parameters: {total_params}")
    print(f"Mismatched parameters: {mismatched_params}")

    if mismatched_params == 0:
        print("Model parameters are very close.")
    else:
        print(f"{mismatched_params} parameter(s) differ by more than the threshold of {threshold}.")
        print("Mismatched parameter differences:")
        for name, diff in model_diff.items():
            print(f"{name}: {diff}")
def compare_models(model_path1, model_path2, threshold=1e-5):
    # Load the model checkpoints
    model1 = torch.load(model_path1, map_location='cpu')
    model2 = torch.load(model_path2, map_location='cpu')
    
    model1_param = model1["model"]
    model2_param = model2["model"]

    # Compare the model parameters
    model_diff = {}
    total_params = 0
    mismatched_params = 0

    for (name1, param1), (name2, param2) in zip(model1_param.items(), model2_param.items()):
        if name1 != name2:
            print(f"Parameter name mismatch: {name1} != {name2}")
            continue
        #transfer the parameters to floating point numbers
        param1 = param1.float()
        param2 = param2.float()
        diff = torch.norm(param1 - param2).item()
        total_params += param1.numel()
        if diff > threshold:
            mismatched_params += 1
            model_diff[name1] = diff

    print(f"Total parameters: {total_params}")
    print(f"Mismatched parameters: {mismatched_params}")

    if mismatched_params == 0:
        print("Model parameters are very close.")
    else:
        print(f"{mismatched_params} parameter(s) differ by more than the threshold of {threshold}.")
        print("Mismatched parameter differences:")
        for name, diff in model_diff.items():
            print(f"{name}: {diff}")

# Usage
model_path1 = '/home/wiss/zhang/Jinhe/singularity/qa_anet/anet_neg_0_from_scratch/ckpt_best.pth'
model_path2 = '/home/wiss/zhang/Jinhe/singularity/qa_anet/anet_neg_0.5_from_scratch/ckpt_best.pth'
compare_models(model_path1, model_path2, threshold=1e-5)
#1e-5