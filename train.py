import turicreate as tc
import coremltools

# configure the GPUs
tc.config.set_num_gpus(0)

# Load SFrame
data = tc.SFrame('/storage/xy_signs.sframe')

# Make a train-test split
train_data, test_data = data.random_split(0.8)
# Create and train model
model = tc.object_detector.create(train_data)
model.evaluate(test_data)
model.export_coreml('/storage/xy_signs.mlmodel')

# reduce model size
model_spec = coremltools.utils.load_spec('/storage/xy_signs.mlmodel')
model_fp16_spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(model_spec)
coremltools.utils.save_spec(model_fp16_spec, '/storage/xy_signs_16bit.mlmodel')