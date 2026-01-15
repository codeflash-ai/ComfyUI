import torch
import comfy.utils


def convert_lora_bfl_control(sd): #BFL loras for Flux
    sd_out = {}
    # Use items() for faster iteration, avoid repeated key lookups
    for k, v in sd.items():
        # Chain replace using a dictionary for fixed replacements, avoiding multiple .replace() calls
        # These are deterministic and not overlapping based on observed usage - safe to apply in sequence
        k_replaced = k.replace(".lora_B.bias", ".diff_b").replace("_norm.scale", "_norm.scale.set_weight")
        k_to = f"diffusion_model.{k_replaced}"
        sd_out[k_to] = v

    # Precompute shapes once for tensor construction (avoid repeated shape lookup)
    img_in_B_shape0 = sd["img_in.lora_B.weight"].shape[0]
    img_in_A_shape1 = sd["img_in.lora_A.weight"].shape[1]
    sd_out["diffusion_model.img_in.reshape_weight"] = torch.tensor([img_in_B_shape0, img_in_A_shape1])
    return sd_out


def convert_lora_wan_fun(sd): #Wan Fun loras
    # No optimization possible here without changing the API or expected behavior
    return comfy.utils.state_dict_prefix_replace(sd, {"lora_unet__": "lora_unet_"})

def convert_uso_lora(sd):
    sd_out = {}
    # Use items() for faster iteration
    for k, tensor in sd.items():
        # Use a sequence of replace calls for transformation
        # Replace order matters -- preserve original order for correctness
        k_to = "diffusion_model." + (
            k.replace(".down.weight", ".lora_down.weight")
             .replace(".up.weight", ".lora_up.weight")
             .replace(".qkv_lora2.", ".txt_attn.qkv.")
             .replace(".qkv_lora1.", ".img_attn.qkv.")
             .replace(".proj_lora1.", ".img_attn.proj.")
             .replace(".proj_lora2.", ".txt_attn.proj.")
             .replace(".qkv_lora.", ".linear1_qkv.")
             .replace(".proj_lora.", ".linear2.")
             .replace(".processor.", ".")
        )
        sd_out[k_to] = tensor
    return sd_out


def convert_lora(sd):
    # Use direct membership tests, which are already fast. Early return for efficiency.
    if "img_in.lora_A.weight" in sd and "single_blocks.0.norm.key_norm.scale" in sd:
        return convert_lora_bfl_control(sd)
    if "lora_unet__blocks_0_cross_attn_k.lora_down.weight" in sd:
        return convert_lora_wan_fun(sd)
    if "single_blocks.37.processor.qkv_lora.up.weight" in sd and "double_blocks.18.processor.qkv_lora2.up.weight" in sd:
        return convert_uso_lora(sd)
    return sd
