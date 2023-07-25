import argparse
from train import train_model
from predict import predict, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Flower Image Classifier Console Application")
    parser.add_argument("data_dir", help="Path to the data directory containing train, valid, and test folders")
    parser.add_argument("save_dir", help="Path to the directory where the model checkpoint will be saved")
    parser.add_argument("--arch", default="vgg16", help="Pretrained model architecture (default: vgg16)")
    parser.add_argument("--hidden_units", type=int, default=512,
                        help="Number of units in the hidden layer (default: 512)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training (default: 5)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training (default: 0.001)")
    parser.add_argument("--topk", type=int, default=5, help="Top K most likely classes for prediction (default: 5)")
    parser.add_argument("--image_path", help="Path to the image for prediction")
    parser.add_argument("--category_names", default="cat_to_name.json",
                        help="Filepath to the JSON file containing category names (default: cat_to_name.json)")

    args = parser.parse_args()

    if args.image_path:
        # Make predictions using the provided image path
        print("Loading the model checkpoint...")
        model, class_to_idx = load_checkpoint(args.save_dir)  # Load the model checkpoint
        print("Model checkpoint loaded successfully.")

        print("Making predictions for the provided image...")
        probs, classes = predict(args.image_path, model, class_to_idx, topk=args.topk)
        print("Predictions:")
        print(probs)
        print(classes)

        for i, (prob, flower_class) in enumerate(zip(probs, classes), 1):
            prob_percent = prob * 100
            print(f"{i}. Class: {flower_class}, Probability: {prob_percent:.2f}%")

    else:
        # Train the model using the data in the data_dir
        train_model(args.data_dir, args.save_dir, args.category_names, arch=args.arch, hidden_units=args.hidden_units,
                    epochs=args.epochs, lr=args.lr)

        print("Training completed. Model checkpoint saved.")


if __name__ == "__main__":
    main()
