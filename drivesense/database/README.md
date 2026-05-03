# Database Layer

DriveSense currently does not use a persistent database.

At the moment, the project is file-based:

- raw datasets live under `dataset/`
- prepared datasets live under `prepared_datasets/`
- model outputs live under `runs_timm/`
- benchmark results are written as CSV/JSON/image files

If the project later needs driver profiles, chat history persistence, or experiment tracking,
this package is the intended place for that storage layer.
