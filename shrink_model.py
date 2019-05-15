import coremltools

# reduce model size
model_spec = coremltools.utils.load_spec('/storage/xy_signs.mlmodel')
model_fp16_spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(model_spec)
coremltools.utils.save_spec(model_fp16_spec, '/storage/xy_signs_16bit.mlmodel')