import argparse
import sys
from pathlib import Path

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    # Resolve default TrainImages path relative to this script
    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir.parents[1] / "TrainImages"
    default_output_plot = script_dir / "class_distribution.png"

    ap = argparse.ArgumentParser(description="Inspect dataset image counts and distribution per class.")
    ap.add_argument("--data-dir", default=str(default_data_dir), help="Path to the TrainImages directory")
    ap.add_argument("--output-plot", default=str(default_output_plot), help="Path to save the distribution plot")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Inspecting dataset in: {data_dir.resolve()}")

    # Gather subdirectories (classes)
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    if not class_dirs:
        print(f"No class subdirectories found in '{data_dir}'.", file=sys.stderr)
        sys.exit(1)

    class_counts = {}
    total_images = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        # Count files with valid extensions
        images = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXT]
        count = len(images)
        class_counts[class_name] = count
        total_images += count

    # Print text summary
    print("\n" + "=" * 45)
    print(f"{'Class Name':<25} | {'Image Count':<15}")
    print("-" * 45)
    for class_name, count in class_counts.items():
        print(f"{class_name:<25} | {count:<15}")
    print("-" * 45)
    print(f"{'Total':<25} | {total_images:<15}")
    print("=" * 45 + "\n")

    # Generate plot using matplotlib
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is not installed. Skipping plot generation.", file=sys.stderr)
        print("Please install matplotlib via: pip install matplotlib", file=sys.stderr)
        sys.exit(0)

    # Set styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    # Beautiful color palette matching the styling guidelines
    colors = ['#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
    colors = colors[:len(classes)]
    if len(colors) < len(classes):
        colors = colors * (len(classes) // len(colors) + 1)
        colors = colors[:len(classes)]

    bars = ax.bar(classes, counts, color=colors, edgecolor='none', width=0.6)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title("Dataset Distribution per Class", fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel("Class Name", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Images", fontsize=12, labelpad=10)
    
    # Adjust layout and save
    plt.tight_layout()
    output_plot_path = Path(args.output_plot)
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    
    print(f"Distribution plot successfully saved to: {output_plot_path.resolve()}")

if __name__ == "__main__":
    main()
