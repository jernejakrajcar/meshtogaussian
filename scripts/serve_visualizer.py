from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Mesh-to-Gaussian web visualizer.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    try:
        import uvicorn
    except Exception as exc:
        raise RuntimeError("Install web dependencies first: pip install -r requirements.txt") from exc

    from src.web.app import create_app

    app = create_app(config_path=args.config, data_dir=args.data_dir)
    print(f"Visualizer running at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
