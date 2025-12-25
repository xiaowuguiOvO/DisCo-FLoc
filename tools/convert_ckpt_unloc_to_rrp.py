import torch
import argparse
import os

def convert_checkpoint(input_path, output_path):
    print(f"Loading checkpoint from: {input_path}")
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return

    # Load on CPU to avoid CUDA errors if running on a machine without GPU
    ckpt = torch.load(input_path, map_location="cpu")
    
    # Checkpoint structure usually has 'state_dict' key for PL models
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        print("Found 'state_dict' in checkpoint.")
    else:
        # If it's a raw PyTorch model save
        state_dict = ckpt
        print("No 'state_dict' key found, assuming raw state dict.")

    new_state_dict = {}
    renamed_count = 0
    
    for key, value in state_dict.items():
        new_key = key
        
        # Replace 'unloc_decoder' with 'rrp_decoder'
        if "unloc_decoder" in key:
            new_key = key.replace("unloc_decoder", "rrp_decoder")
            print(f"Renaming: {key} -> {new_key}")
            renamed_count += 1
            
        # Add other replacements here if needed in the future
        # e.g., if you renamed other modules
        
        new_state_dict[new_key] = value

    # Update state_dict in the checkpoint
    if "state_dict" in ckpt:
        ckpt["state_dict"] = new_state_dict
    else:
        ckpt = new_state_dict

    print(f"Total keys renamed: {renamed_count}")

    # Update hyper_parameters if they exist
    if "hyper_parameters" in ckpt:
        print("Found 'hyper_parameters' in checkpoint. Checking for 'decoder_type'...")
        hparams = ckpt["hyper_parameters"]
        
        # Check inside 'config' dict if it exists (common pattern)
        if "config" in hparams and isinstance(hparams["config"], dict):
            config = hparams["config"]
            if config.get("decoder_type") == "unloc":
                print("Updating config['decoder_type']: 'unloc' -> 'rrp'")
                config["decoder_type"] = "rrp"
        
        # Check top-level decoder_type
        if hparams.get("decoder_type") == "unloc":
            print("Updating hparams['decoder_type']: 'unloc' -> 'rrp'")
            hparams["decoder_type"] = "rrp"
            
    # Save the new checkpoint
    print(f"Saving new checkpoint to: {output_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(ckpt, output_path)
    print("Conversion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert UnLoc checkpoint to RRP checkpoint.")
    parser.add_argument("--input_ckpt", type=str, required=True, help="Path to the input (old) checkpoint file.")
    parser.add_argument("--output_ckpt", type=str, required=True, help="Path to save the output (new) checkpoint file.")
    
    args = parser.parse_args()
    
    convert_checkpoint(args.input_ckpt, args.output_ckpt)
