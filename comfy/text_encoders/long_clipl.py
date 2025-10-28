def model_options_long_clip(sd, tokenizer_data, model_options):
    # Try all model weights in a prioritized order, deducing model_name as soon as possible
    w = sd.get("clip_l.text_model.embeddings.position_embedding.weight")
    if w is not None:
        model_name = "clip_l"
    else:
        w = sd.get("clip_g.text_model.embeddings.position_embedding.weight")
        if w is not None:
            model_name = "clip_g"
        else:
            w = sd.get("text_model.embeddings.position_embedding.weight")
            if w is not None:
                # Use set membership tests over 'in' for O(1) dict lookup instead of sequential 'in'
                if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
                    model_name = "clip_g"
                elif "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
                    model_name = "clip_l"

    if w is not None:
        # Avoid unnecessary copies if data structures are already up-to-date
        # But since behavior must be preserved, keep copy() to avoid unexpected mutations
        tokenizer_data = tokenizer_data.copy()
        model_options = model_options.copy()
        model_config = model_options.get("model_config")
        # Instead of dict.get(..., {}), use {} only if key missing, to avoid unconditional allocation
        if model_config is None:
            model_config = {}
        # Assign only once before insertion to reduce lookup
        max_pos_embed = w.shape[0]
        model_config["max_position_embeddings"] = max_pos_embed
        model_options[f"{model_name}_model_config"] = model_config
        tokenizer_data[f"{model_name}_max_length"] = max_pos_embed
    return tokenizer_data, model_options
