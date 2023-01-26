from torchvision import transforms


class MyTransformer(object):
    def __init__(self):
        self.transformer = transforms.Compose([
            transforms.ToTensor(), # from PIL to tensor
            transforms.Lambda(lambda x: x.permute(2, 0, 1)), # This will reorder the dimension of the tensor from (height, width, channel) to (channel, height, width)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #normalize pixel image, subtract the mean and divide by std
        ])

    def __call__(self, x):
        return self.transformer(x)


my_transformer = MyTransformer()
