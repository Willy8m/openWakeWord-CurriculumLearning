import os
import tempfile

import numpy as np
import scipy

import openwakeword

with tempfile.TemporaryDirectory() as tmp_dir:
    # Make random negative data for verifier model training
    scipy.io.wavfile.write(os.path.join(tmp_dir, "negative_reference.wav"),
                           16000, np.random.randint(-1000, 1000, 16000 * 4).astype(np.int16))

    # Load random clips
    reference_clips = [os.path.join("data", "alexa_test.wav")]
    negative_clips = [os.path.join(tmp_dir, "negative_reference.wav")]

    # Check for error message when no positive examples are found

    openwakeword.train_custom_verifier(
        positive_reference_clips=reference_clips,
        negative_reference_clips=negative_clips,
        output_path=os.path.join('verifier_model.pkl'),
        model_name="alexa",
        inference_framework="onnxruntime"
    )

    print(12)
