# Contributing

Please include the exact command, Python/PyTorch/Transformers versions, hardware and a small reproducer when reporting a problem. Use synthetic or redistributable inputs; do not attach credentials, model weights or private media.

Keep historical evidence immutable. Add a separate experiment or validation record when changing behavior. Claims about accuracy, speed or memory need a clearly matched comparison and a stated measurement boundary. CPU-only fixes are welcome; a failed numerical check should not be addressed by silently loosening its tolerance.

Run the commands in the README and the unittest suite before submitting a change. New model or runtime support should state which cases were actually tested.
