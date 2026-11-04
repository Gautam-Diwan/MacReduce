from macreduce.plots import write_run_figures


def test_run_figures_cover_loss_energy_memory_and_latency(tmp_path):
    history = {
        "train": [
            {
                "iteration": 10,
                "train_loss": 4.2,
                "iterations_per_second": 2.0,
                "tokens_per_second": 400.0,
                "peak_memory": 1.2,
            },
            {
                "iteration": 20,
                "train_loss": 3.8,
                "iterations_per_second": 2.5,
                "tokens_per_second": 500.0,
                "peak_memory": 1.4,
            },
        ],
        "validation": [
            {"iteration": 10, "val_loss": 4.0, "val_perplexity": 54.6, "val_time": 1.2},
            {"iteration": 20, "val_loss": 3.7, "val_perplexity": 40.4, "val_time": 1.1},
        ],
    }
    samples = [
        {"t": 1000.0, "gpu": 10.0, "ane": 0.0, "memory_used_bytes": 8e9},
        {"t": 1001.0, "gpu": 12.0, "ane": 0.4, "memory_used_bytes": 9e9},
        {"t": 1002.0, "gpu": 11.0, "ane": 0.5, "memory_used_bytes": 9.2e9},
    ]
    written = write_run_figures(tmp_path, history, samples)
    names = {path.name for path in written}
    assert names == {
        "loss.png",
        "perplexity.png",
        "energy.png",
        "memory.png",
        "latency.png",
    }
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in written)