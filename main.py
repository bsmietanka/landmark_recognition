# TODO: more verbose name of file?

import argparse
from model import model

def main():
    parser = argparse.ArgumentParser(description='Deep neural network model for landmark recognition')
    parser.add_argument('dataset_path', metavar='dataset_path', nargs='?', default='', help='path to a dataset directory')
    # read dataset from csv and download to local directiory?
    parser.add_argument('-m', '--model',
                        default="DenseNet121",
                        choices=model.types,
                        help='indicates which model should be used')
    parser.add_argument('-p', '--predict', help='path to photo to predict landmark')
    parser.add_argument('-l', '--load', help='path to saved model weights')
    parser.add_argument('-f', '--freeze', action='store_true', help='freeze top convolutional layers when using pretrained weights')
    parser.add_argument('-v', '--validate', action='store_true', help='validation directory')
    args = parser.parse_args()

    my_model = model()

    if args.predict is not None and args.load is None:
        print("Provide saved model path to predict landmark")
        return

    if args.load is not None:
        my_model.load_model(args.load)
    elif args.dataset_path is not None:
        my_model.prepare_data_generators(args.dataset_path, args.validate)
        my_model.instantiate_model(args.model, args.freeze)
    else:
        print("Provide path to dataset to train or validate instantiated model")
        return
    
    if args.validate:
        my_model.validate()
    elif args.predict is not None:
        my_model.predict(args.predict)
    else:
        my_model.train()

if __name__ == "__main__":
    main()