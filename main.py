# main.py
"""
CLI entry point for StoneX.

Usage examples:

  # Ingest full dataset (colour + embedding)
  python main.py ingest /path/to/dataset

  # Ingest colour only
  python main.py ingest /path/to/dataset --color-only

  # Ingest embedding only
  python main.py ingest /path/to/dataset --embedding-only

  # Query with default pipeline (color → embedding → model)
  python main.py query /path/to/image.jpg

  # Query with custom pipeline
  python main.py query /path/to/image.jpg --layers model embedding color

  # Query — show top 5 families
  python main.py query /path/to/image.jpg --top-k 5
"""

import argparse
import sys


def cmd_ingest(args):
    from ingestion.ingest_dataset import ingest_dataset
    do_color     = not args.embedding_only
    do_embedding = not args.color_only
    ingest_dataset(args.folder, do_color=do_color, do_embedding=do_embedding)


def cmd_query(args):
    from query.pipeline import run_pipeline

    layer_order = args.layers or ["color", "embedding", "model"]
    print(f"\n🔍 Pipeline: {' → '.join(layer_order)}")
    print(f"   Image   : {args.image}")
    print(f"   Top-K   : {args.top_k}\n")

    results = run_pipeline(
        args.image,
        layer_order=layer_order,
        top_k_families=args.top_k,
        top_k_images=args.top_k_images,
        first_layer_fetch=args.first_layer_fetch,
    )

    print("🏆 Top Stone Families:")
    for i, (family, score) in enumerate(results["families"], 1):
        bar = "█" * int(score * 30)
        print(f"  {i:2}. {family:<40} {score:.4f}  {bar}")

    if args.show_images:
        print("\n🖼️  Top matched images:")
        for layer in layer_order:
            imgs = results["images"].get(layer, [])
            if imgs:
                print(f"\n  [{layer}]")
                for path, score in imgs[:5]:
                    print(f"    {score:.4f}  {path}")


def main():
    parser = argparse.ArgumentParser(prog="stonex", description="StoneX — Stone family search")
    sub = parser.add_subparsers(dest="command")

    # ── ingest ──
    p_ingest = sub.add_parser("ingest", help="Ingest a dataset folder")
    p_ingest.add_argument("folder", help="Path to the parent dataset folder")
    p_ingest.add_argument("--color-only",     action="store_true")
    p_ingest.add_argument("--embedding-only", action="store_true")

    # ── query ──
    p_query = sub.add_parser("query", help="Query with an image")
    p_query.add_argument("image", help="Path to the query image")
    p_query.add_argument("--layers", nargs="+", choices=["color", "embedding", "model"],
                         default=None, help="Layer order (default: color embedding model)")
    p_query.add_argument("--top-k",           type=int, default=10)
    p_query.add_argument("--top-k-images",    type=int, default=10)
    p_query.add_argument("--first-layer-fetch", type=int, default=40)
    p_query.add_argument("--show-images",     action="store_true")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()