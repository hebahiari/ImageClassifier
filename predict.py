from get_input_args import get_predict_args
from model_utils import load_checkpoint, predict
from data_utils import process_image

def main():
    args = get_predict_args()

    # Load the model checkpoint
    model = load_checkpoint(args.checkpoint)

    # Preprocess the input image
    image = process_image(args.image_path)

    # Predict the class
    probs, classes = predict(image, model, args.top_k, args.gpu)

    # Display the results
    print(f"Top {args.top_k} Classes:")
    for prob, cls in zip(probs, classes):
        print(f"Class: {cls}, Probability: {prob}")

if __name__ == "__main__":
    main()
