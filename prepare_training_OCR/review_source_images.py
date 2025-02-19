import json
from collections import defaultdict
from datasets import load_dataset
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import json
from collections import defaultdict
from datasets import load_dataset
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def reconstruct_source_image(examples):
    """
    Given a list of training examples (lines) from the same 'source_image',
    sort them by 'y1' and vertically concatenate the line images (with text) 
    to approximate the original page.
    """
    # Sort examples from top to bottom
    sorted_examples = sorted(examples, key=lambda ex: ex['y1'])
    
    # Compute required canvas size (add spacing between lines)
    max_width = max(example['image'].width for example in sorted_examples)
    total_height = sum(example['image'].height for example in sorted_examples) + (len(sorted_examples) - 1) * 20
    
    # Create a blank white canvas
    combined_img = Image.new('RGB', (max_width, total_height), color=(255, 255, 255))
    y_offset = 0
    for example in sorted_examples:
        line_img = example['image']
        combined_img.paste(line_img, (0, y_offset))
        # Draw the text below the line image
        draw = ImageDraw.Draw(combined_img)
        text_position = (10, y_offset + line_img.height + 5)
        draw.text(text_position, example['text'], fill="black")
        y_offset += line_img.height + 20
    return combined_img

def main():
    # Load the dataset (this may take a moment)
    dataset = load_dataset("agomberto/handwritten-ocr-dataset")
    train_samples = dataset['train']
    
    # Group the training examples by "source_image"
    groups = defaultdict(list)
    for sample in train_samples:
        groups[sample['source_image']].append(sample)
        
    print("Total distinct source images:", len(groups))
    
    # Get the list of source image identifiers
    source_images = list(groups.keys())
    n_images = len(source_images)
    
    # Try to resume from previous decisions if available.
    decisions_file = "image_decisions.json"
    try:
        with open(decisions_file, "r") as f:
            decisions = json.load(f)
    except FileNotFoundError:
        decisions = {}
    
    # Enable interactive matplotlib mode and create a persistent figure
    plt.ion()
    fig, ax = plt.subplots()
    
    # Iterate through each source image that hasn't been processed yet.
    for idx, src_id in enumerate(source_images):
        if src_id in decisions:
            print(f"Skipping {src_id} (already processed)")
            continue
        
        print(f"\nProcessing image {idx+1}/{n_images}: {src_id}")
        combined = reconstruct_source_image(groups[src_id])
        
        # Update the existing figure with the new image
        # Set a fixed width for the figure (e.g., 16 inches); compute height by maintaining the aspect ratio.
        desired_width = 16
        aspect_ratio = combined.height / combined.width
        fig.set_size_inches(desired_width, desired_width * aspect_ratio)
        
        # Force the figure window to the computed dimensions (works for TkAgg backend)
        dpi = fig.get_dpi()
        new_width_px = int(desired_width * dpi)
        new_height_px = int(desired_width * aspect_ratio * dpi)
        try:
            mng = plt.get_current_fig_manager()
            mng.window.wm_geometry(f"{new_width_px}x{new_height_px}")
        except Exception as e:
            print("Could not resize figure window:", e)
        
        ax.clear()
        ax.imshow(combined)
        # Reset axis limits to display the full image regardless of prior zoom/pan.
        ax.set_xlim(0, combined.width)
        ax.set_ylim(combined.height, 0)
        ax.set_title(f"Source Image: {src_id}\nTotal lines: {len(groups[src_id])}", fontsize=12)
        ax.axis("off")
        plt.draw()
        plt.pause(0.001)  # short pause to update the figure
        
        # Get user decision: A=Accept, R=Reject, or E=Exit early
        while True:
            user_input = input("Accept this image? (A)ccept / (R)eject / (E)xit: ").strip().lower()
            if user_input == "a":
                decisions[src_id] = {"choice": "accepted", "num_lines": len(groups[src_id])}
                break
            elif user_input == "r":
                decisions[src_id] = {"choice": "rejected", "num_lines": len(groups[src_id])}
                break
            elif user_input == "e":
                with open(decisions_file, "w") as f:
                    json.dump(decisions, f, indent=4)
                print("Exiting early. Decisions saved to", decisions_file)
                return
            else:
                print("Invalid input. Please enter A, R, or E.")
        
        # Save progress after each decision
        with open(decisions_file, "w") as f:
            json.dump(decisions, f, indent=4)
            
    print("\nReview complete. Decisions saved to", decisions_file)

if __name__ == "__main__":
    main() 