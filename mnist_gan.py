import torch

class Generator(torch.nn.Module):
    """
    A generator class.
    Accepts a noise tensor of shape (batch_size, code_size),
    outputs a batch of images of shape (batch_size, 1, 28, 28)
    
    Reuses architecture from https://www.tensorflow.org/tutorials/generative/dcgan
    """
    def __init__(self, code_size):
        super(Generator, self).__init__()
        self.code_size = code_size
        # A single dense layer with batch normalisation and leaky ReLU activation
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(self.code_size, 7*7*256, bias=False),
            torch.nn.BatchNorm1d(7*7*256),
            torch.nn.LeakyReLU()
            )
        # Deconvolution layers that increase the image resolution
        # and decrease the number of channels
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1,
                                     bias=False, padding=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, bias=False,
                                     padding=2, output_padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(), 
            
            torch.nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, bias=False,
                                     padding=2, output_padding=1),
            torch.nn.Tanh()
            )

    def forward(self, input):
        x = self.dense(input)
        x = x.view(-1, 256, 7, 7)
        return self.main(x)

class Discriminator(torch.nn.Module):
    """
    Discriminator class.
    
    Input: a batch of images of shape (batch_size, 1, 28, 28)
    Output: a batch of predictions of shape (batch_size)
    
    Reuses architecture from https://www.tensorflow.org/tutorials/generative/dcgan
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_part = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2), # 14*14
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Conv2d(64, 128, 5, 2, padding=2), # 7*7
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.3),
        )
        self.head = torch.nn.Linear(128*7*7, 1)

    def forward(self, input):
        x = self.conv_part(input)
        x = x.view(-1, 128*7*7)
        return self.head(x).view(-1)
