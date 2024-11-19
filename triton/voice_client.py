import numpy as np
import json
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import torchaudio

# Function to load and preprocess audio
def load_audio(audio_file_path, target_sample_rate=16000):
    waveform, sample_rate = torchaudio.load(audio_file_path)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    # Assuming the model expects a 2D tensor as input; adjust as necessary
    audio_data = waveform.numpy().astype(np.float32)
    return audio_data

# Function to send a request to Triton Inference Server and receive the response
def infer(audio_file_path, model_name="audio_model", server_url="localhost:8000"):
    # Load and preprocess the audio file
    audio_data = load_audio(audio_file_path)
    
    # Flatten the audio data to a 1D array for HTTP/REST request
    
    # Create Triton client
    try:
        triton_client = httpclient.InferenceServerClient(url=server_url, verbose=True)
    except Exception as e:
        print(f"Failed to connect to Triton Inference Server at {server_url}: {e}")
        return

    # Prepare the input tensor
    inputs = []
    inputs.append(httpclient.InferInput("AUDIO_INPUT", audio_data.shape, "FP32"))
    inputs[0].set_data_from_numpy(audio_data)
    
    # Prepare the output tensor
    outputs = []
    outputs.append(httpclient.InferRequestedOutput("OUTPUT_DICT"))
    
    # Send inference request to the server and get response
    try:
        response = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        result = response.get_response()
        print("Inference result:", result)
        
        # Decode the output
        output_data = response.as_numpy("OUTPUT_DICT")
        if output_data is not None:
            print("Decoded output:", json.loads(output_data[0].decode('utf-8')))
            
        else:
            print("Failed to retrieve output data.")
            
    except InferenceServerException as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    audio_file_path = "10_utts.wav"  # Update this path to your audio file
    infer(audio_file_path)
