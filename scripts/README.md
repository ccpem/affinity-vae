# Useful scripts for AffinityVAE

## Napari viewer for trained models

`run_napari_model_view.py` lets you inspect the latent space and decoder of a
trained model. Setup (a separate virtual environment, because Napari needs
`pydantic<2`), options, widgets and troubleshooting are documented in the main
README: [Inspecting results with Napari](../README.md#inspecting-results-with-napari).

Quick reference:

```bash
python scripts/run_napari_model_view.py --help
python scripts/run_napari_model_view.py --model_file <path_to_model> --meta_file <path_to_meta> [--manifold umap|load]
```
