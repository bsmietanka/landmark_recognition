# TODO: more verbose name of file?

import argparse
from model import model
import numpy as np
from sklearn import metrics
from os.path import dirname, join


def main():
    parser = argparse.ArgumentParser(description='Deep neural network model for landmark recognition')
    parser.add_argument('dataset_path', metavar='dataset_path', help='path to a dataset directory')
    # TODO: read dataset from csv and download to local directiory?
    parser.add_argument('-m', '--model',
                        default="DenseNet121",
                        choices=model.types,
                        help='indicates which model should be used')
    parser.add_argument('-p', '--predict', action='store_true', help='path to photo to predict landmark')
    parser.add_argument('-l', '--load', help='path to saved model weights')
    parser.add_argument('-f', '--freeze', action='store_true', help='freeze top convolutional layers when using pretrained weights')
    parser.add_argument('-v', '--validate', action='store_true', help='validation directory')
    args = parser.parse_args()

    my_model = model()

    if args.predict and args.load is None:
        print("Provide saved model path to predict landmark")
        return

    my_model.prepare_data_generators(args.dataset_path, args.validate or args.predict)
    if args.load is not None:
        my_model.load_model(args.load)
    elif args.dataset_path is not None:
        my_model.instantiate_model(args.model, args.freeze)
    else:
        print("Provide path to dataset to train or validate instantiated model")
        print("For more information on usage type --help")
        return
    
    if args.validate:
        print(my_model.validate())
    elif args.predict:
        test_data_generator = my_model.validation_generator
        true_classes = test_data_generator.classes
        class_labels = list(test_data_generator.class_indices.keys())
        predictions = my_model.predict()
        predicted_classes = np.argmax(predictions, axis=1)
        report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
        print(report)
        with open(join(dirname(args.load), "report.txt"), "w") as text_file:
            print(f"{report}", file=text_file)
    else:
        my_model.train()


if __name__ == "__main__":
    main()