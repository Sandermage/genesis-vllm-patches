# SGLang pin manifests

One directory per validated SGLang pin, each containing a `manifest.yaml`
(file → md5 + anchor map), mirroring `sndr/engines/vllm/pins/`.

`SglangEngine.list_supported_pins()` enumerates the subdirs here that contain
a `manifest.yaml`; `is_pin_supported(pin)` checks one. Empty until the first
SGLang pin is validated and a manifest is generated (`tools/manifest_gen.py`).
