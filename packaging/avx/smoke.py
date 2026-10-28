#!/usr/bin/env python3
"""Model-free CPU operations, autograd, serialization and native inference smoke."""
import json
import os
from pathlib import Path
import resource
import tempfile


def check(scratch):
    import numpy as np
    import paddle

    if paddle.is_compiled_with_cuda():
        raise RuntimeError("Install the CPU-only release wheel before running this check")
    paddle.set_device("cpu")
    paddle.disable_static()
    paddle.seed(123)
    passed = []
    with paddle.base.dygraph.guard(paddle.CPUPlace()):
        values = np.arange(16, dtype="float32").reshape(4, 4) / 8
        x = paddle.to_tensor(values)
        np.testing.assert_allclose((x @ x).numpy(), values @ values, rtol=1e-5)
        passed.append("matmul")
        conv = paddle.nn.Conv2D(1, 2, 3)
        assert tuple(conv(paddle.ones((1, 1, 8, 8))).shape) == (1, 2, 6, 6)
        passed.append("convolution")
        for dtype in ("float32", "float64"):
            array = np.arange(16, dtype=dtype) / 8
            real = paddle.fft.rfft(paddle.to_tensor(array)).numpy()
            np.testing.assert_allclose(real, np.fft.rfft(array), rtol=1e-5, atol=1e-5)
            restored = paddle.fft.irfft(paddle.to_tensor(real), n=16).numpy()
            np.testing.assert_allclose(restored, array, rtol=1e-5, atol=1e-5)
            complex_type = "complex64" if dtype == "float32" else "complex128"
            complex_values = array.astype(complex_type) + 1j * array[::-1]
            transformed = paddle.fft.fft(paddle.to_tensor(complex_values)).numpy()
            np.testing.assert_allclose(transformed, np.fft.fft(complex_values), rtol=1e-5, atol=1e-5)
            passed.append("fft-" + dtype)
        grad = paddle.to_tensor(np.array([1., 2., 3.], dtype="float32"), stop_gradient=False)
        paddle.sum(grad * grad).backward()
        np.testing.assert_allclose(grad.grad.numpy(), [2., 4., 6.])
        passed.append("autograd")
        net = paddle.nn.Linear(4, 2)
        net.eval()
        with paddle.no_grad():
            expected = net(x).numpy()
            checkpoint = scratch / "state.pdparams"
            paddle.save(net.state_dict(), str(checkpoint))
            restored = paddle.nn.Linear(4, 2)
            restored.set_state_dict(paddle.load(str(checkpoint)))
            restored.eval()
            np.testing.assert_allclose(restored(x).numpy(), expected, rtol=1e-5)
            passed.append("state-roundtrip")
            model = str(scratch / "model")
            exported = paddle.jit.to_static(net, input_spec=[paddle.static.InputSpec([None, 4], "float32")], full_graph=True)
            paddle.jit.save(exported, model)
    from paddle.inference import Config, create_predictor
    config = Config(model + ".json", model + ".pdiparams")
    config.disable_gpu()
    config.disable_mkldnn()
    config.disable_glog_info()
    config.switch_ir_optim(False)
    config.set_cpu_math_library_num_threads(1)
    predictor = create_predictor(config)
    handle = predictor.get_input_handle(predictor.get_input_names()[0])
    handle.reshape(values.shape)
    handle.copy_from_cpu(values)
    predictor.run()
    result = predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    passed.append("export-native-inference")
    print(json.dumps(dict(paddle_version=paddle.__version__, device="cpu", passed=passed), indent=2))


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    with tempfile.TemporaryDirectory(prefix="paddle-avx-smoke-") as directory:
        scratch = Path(directory)
        os.environ.update(TMPDIR=str(scratch), PADDLE_HOME=str(scratch / "paddle"),
                          XDG_CACHE_HOME=str(scratch / "cache"), OMP_NUM_THREADS="1",
                          OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        # The legacy dataset import uses tilde expansion rather than PADDLE_HOME.
        expanduser = os.path.expanduser
        def isolated_expanduser(path):
            value = os.fsdecode(path)
            if value == "~" or value.startswith("~/"):
                target = str(scratch / "user" / value[2:])
                return os.fsencode(target) if isinstance(path, bytes) else target
            return expanduser(path)
        os.path.expanduser = isolated_expanduser
        try:
            check(scratch)
        finally:
            os.path.expanduser = expanduser


if __name__ == "__main__":
    main()
