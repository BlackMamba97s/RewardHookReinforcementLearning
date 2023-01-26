from torchvision import transforms


class MyTransformer(object):
    def __init__(self):
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.permute(2, 0, 1)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __call__(self, x):
        return self.transformer(x)


my_transformer = MyTransformer()
