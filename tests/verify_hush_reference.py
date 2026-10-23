import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import wave

import numpy as np
import onnxruntime as ort


def read_audio(path):
    with wave.open(str(path)) as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
            raise ValueError("input must be 16 kHz mono PCM16")
        return np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768


def write_audio(path, audio):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 16000, len(audio), "NONE", "not compressed"))
        output.writeframes((np.clip(audio, -1, .999969) * 32768).astype("<i2").tobytes())


def read_dump(directory, graph):
    tensors = {}
    for row in (directory / (graph + ".tsv")).read_text().splitlines():
        name, filename, dimensions = row.split("\t")
        shape = tuple(int(size) for size in dimensions.split(",") if size)
        tensors[name] = np.fromfile(directory / filename, dtype=np.float32).reshape(shape)
    return tensors


def frontend(audio):
    count = (len(audio) + 159) // 160
    window = np.sin(np.pi / 2 * np.sin(np.pi * (np.arange(320) + .5) / 320) ** 2)
    padded = np.pad(audio, (160, count * 160 - len(audio)))
    spectrum = np.stack([np.fft.rfft(padded[time * 160:time * 160 + 320] * window) / 320
                         for time in range(count)]).astype(np.complex64)
    high = 9.265 * np.log1p(8000 / (24.7 * 9.265))
    edges = 24.7 * 9.265 * np.expm1(np.linspace(0, high, 33) / 9.265)
    widths, previous, carried = [], 0, 0
    for edge in edges[1:]:
        frequency = int(np.floor(edge / 50 + .5))
        width = frequency - previous - carried
        carried = max(0, 2 - width)
        widths.append(max(width, 2))
        previous = frequency
    widths[-1] += 161 - sum(widths)
    erb = np.empty((count, 32), np.float32)
    offset = 0
    for band, width in enumerate(widths):
        erb[:, band] = 10 * np.log10(np.mean(abs(spectrum[:, offset:offset + width]) ** 2, axis=1) + 1e-10)
        offset += width
    df = spectrum[:, :64].copy()
    erb_state = np.linspace(-60, -90, 32, dtype=np.float32)
    df_state = np.linspace(.001, .0001, 64, dtype=np.float32)
    for time in range(count):
        erb_state = .99 * erb_state + .01 * erb[time]
        erb[time] = (erb[time] - erb_state) / 40
        df_state = .99 * df_state + .01 * abs(df[time])
        df[time] /= np.sqrt(df_state)
    return {"feat_erb": erb[None, None], "feat_spec": np.stack((df.real, df.imag))[None]}, spectrum, widths, window


def infer(sessions, inputs):
    encoder = dict(zip([item.name for item in sessions[0].get_outputs()], sessions[0].run(None, inputs)))
    erb = sessions[1].run(None, {item.name: encoder[item.name] for item in sessions[1].get_inputs()})[0]
    df = sessions[2].run(None, {item.name: encoder[item.name] for item in sessions[2].get_inputs()})[0]
    return encoder, {"m": erb}, {"coefs": df}


def reconstruct(spectrum, masks, coefs, widths, window, samples):
    count = len(spectrum)
    enhanced = spectrum * np.repeat(masks.reshape(count, 32), widths, axis=1)
    history = np.pad(spectrum[:, :64], ((4, 0), (0, 0)))
    coefficients = coefs.reshape(count, 64, 5, 2)
    coefficients = coefficients[..., 0] + 1j * coefficients[..., 1]
    enhanced[:, :64] = sum(history[tap:tap + count] * coefficients[:, :, tap] for tap in range(5))
    output = np.zeros((count + 1) * 160)
    for time in range(count):
        output[time * 160:time * 160 + 320] += np.fft.irfft(enhanced[time], n=320) * 320 * window
    return output[:samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    input_path = args.input
    if input_path is None:
        input_path = args.workdir / "synthetic.wav"
        time = np.arange(34001) / 16000
        audio = .1 * np.sin(2 * np.pi * 173 * time) * (.6 + .4 * np.sin(2 * np.pi * 3 * time))
        audio += .02 * np.random.default_rng(42).standard_normal(len(time))
        write_audio(input_path, audio)
    audio = read_audio(input_path)
    features, spectrum, widths, window = frontend(audio)
    sessions = []
    with tarfile.open(args.bundle) as archive:
        for filename in ("enc.onnx", "erb_dec.onnx", "df_dec.onnx"):
            model = archive.extractfile(filename).read()
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            sessions.append(ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"]))
    _, erb, df = infer(sessions, features)
    reference = reconstruct(spectrum, erb["m"], df["coefs"], widths, window, len(audio))
    write_audio(args.workdir / "onnxruntime-reference.wav", reference)
    results = {"model_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(), "samples": len(audio), "runs": []}
    for backend in ("cpu", "metal"):
        for batch_size in (1, 32):
            directory = args.workdir / (backend + "-" + str(batch_size))
            environment = dict(os.environ, HUSH_MODEL_BUNDLE=str(args.bundle.resolve()), HUSH_TEST_BATCH_SIZE=str(batch_size))
            environment.pop("HUSH_BACKEND", None)
            if backend == "metal":
                environment["HUSH_BACKEND"] = "metal"
            subprocess.run([str(args.probe.resolve()), str(input_path.resolve()), str(directory.resolve())], env=environment, check=True)
            actual_features = read_dump(directory, "features")
            feature_error = max(float(np.max(abs(actual_features[name] - values))) for name, values in features.items())
            assert feature_error < 2e-4, (backend, batch_size, "frontend", feature_error)
            graph_inputs = {name: values[:, :, :32, :] for name, values in actual_features.items() if name.startswith("feat")}
            graph_error = 0.0
            for graph, expected in zip(("enc", "erb", "df"), infer(sessions, graph_inputs)):
                actual = read_dump(directory, graph)
                for name, value in expected.items():
                    assert actual[name].shape == value.shape, (graph, name, actual[name].shape, value.shape)
                    graph_error = max(graph_error, float(np.max(abs(actual[name] - value))))
            assert graph_error < 2e-4, (backend, batch_size, "graph", graph_error)
            actual_audio = read_dump(directory, "audio")["output"]
            assert actual_audio.shape == reference.shape
            sample_error = float(np.max(abs(actual_audio - reference)))
            assert sample_error < 5e-5, (backend, batch_size, "audio", sample_error)
            pcm = read_audio(directory / "output.wav")
            snr = float(10 * np.log10(np.sum(reference ** 2) / np.sum((pcm - reference) ** 2)))
            assert snr > 50, (backend, batch_size, "PCM reference SNR", snr)
            result = dict(backend=backend, batch_size=batch_size, feature_max_error=feature_error,
                          graph_max_error=graph_error, sample_max_error=sample_error, pcm_reference_snr_db=snr)
            results["runs"].append(result)
            print(json.dumps(result), flush=True)
    (args.workdir / "report.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
