from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator, Iterator

class VariableImageDataGenerator(DirectoryIterator):
    def __init__(self, image_data_generator, sizes, path, batch_size, class_mode):
        self.i = 0
        self.generators = []
        self.num_gen = len(sizes)
        self.sizes = sizes
        for size in sizes:
            self.generators.append(
                image_data_generator.flow_from_directory(
                    path, target_size=size, batch_size=batch_size, class_mode=class_mode
                )
            )
        if len(sizes) > 0:
            self.num_samples = len(self.generators[0])
            self.class_indices = self.generators[0].class_indices
            self.num_classes = self.generators[0].num_classes
            self.classes = self.generators[0].classes
        
    def next(self):
        gen = self.generators[self.i]
        self.image_shape = self.sizes[self.i]
        self.i = (self.i + 1) % self.num_gen
        with gen.lock:
            index_array = next(gen.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return gen._get_batches_of_transformed_samples(index_array)

    def generate(self):
        while True:
            for i in range(0, self.num_gen):
                yield self.generators[i].next()

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    v = VariableImageDataGenerator(
        train_datagen,
        [(200, 200), (400, 400)],
        "dataset",
        8,
        "categorical")

    i = 0
    for (x, y) in v.generate():
        i += 1
        if i >= 10:
            break
        # x = v.generate()
        # print(f"X : {x.shape}")
        print(f"X : {x}")
        # print(f"Y : {y}")
