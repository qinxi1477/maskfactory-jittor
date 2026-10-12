import os
import argparse
import jittor as jt
from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.jittor_compat import enable_cuda, interpolate, read_image, save_image

# Function to load and prepare the input image for inference
def load_and_prepare_image(image_path):
    """
    Load and prepare an image from a given path.
    """
    image = read_image(image_path).float() / 255.0  # Normalize to [0, 1]
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)  # Convert grayscale to RGB
    image = interpolate(image.unsqueeze(0), (512, 512))  # Resize the image
    return image

# Function for inference to generate images
def infer(generator_pipeline, image_path, output_dir):
    """
    Perform inference and generate new images.
    """
    generator_pipeline.eval()  # Set the generator pipeline to evaluation mode
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare the image
    image = load_and_prepare_image(image_path)

    # Generate image using the generator pipeline
    with jt.no_grad():
        generated_image = generator_pipeline(image)

    # Save the generated image
    save_image(generated_image * 0.5 + 0.5, os.path.join(output_dir, 'generated_image.png'))
    print(f"Generated image saved at {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_dir", default="output_images")
    return parser.parse_args()


# Define the main function for inference
def main():
    enable_cuda()
    args = parse_args()

    # Load the pretrained generator pipeline
    generator_pipeline = MasaCtrlPipeline.from_pretrained(args.model_path)

    # Perform inference
    infer(generator_pipeline, args.image_path, args.output_dir)

if __name__ == "__main__":
    main()
