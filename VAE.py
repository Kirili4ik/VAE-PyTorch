# The code is taken from https://github.com/ku2482/vae.pytorch

class Encoder(nn.Module):
    def __init__(self, num_ch, nef, z_size, im_size, device):
        super(Encoder, self).__init__()

        # Device
        self.device = device

        # Encoder: (num_ch, im_size, im_size) -> (nef*8, im_size//16, im_size//16)
        self.encoder = nn.Sequential(
            nn.Conv2d(num_ch, nef, 3, 2, padding=1),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(0.2),

            nn.Conv2d(nef, nef*2, 3, 2, padding=1),
            nn.BatchNorm2d(nef*2),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(nef*2, nef*4, 3, 2, padding=1),
            nn.BatchNorm2d(nef*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(nef*4, nef*8, 3, 2, padding=1),
            nn.BatchNorm2d(nef*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(nef*8, nef*16, 3, 2, padding=1),
            nn.BatchNorm2d(nef*16),
            nn.LeakyReLU(0.2)
        )

        # Map the encoded feature map to the latent vector of mean, (log)variance
        out_size = im_size // 32
        self.mean = nn.Linear(nef * 16 * out_size * out_size, z_size)
        self.logvar = nn.Linear(nef * 16 * out_size * out_size, z_size)

    @staticmethod
    def reparametrize(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, inputs):
        # Batch size
        batch_size = inputs.size(0)
        # Encoded feature map
        hidden = self.encoder(inputs)
        # Reshape
        hidden = hidden.view(batch_size, -1)
        #print('hid', hidden.size())
        # Calculate mean and (log)variance
        mean, logvar = self.mean(hidden), self.logvar(hidden)
        # Sample
        latent_z = self.reparametrize(mean, logvar)

        return latent_z, mean, logvar


class Decoder(nn.Module):
    def __init__(self, num_ch, ndf, z_size, im_size):
        super(Decoder, self).__init__()

        # Map the latent vector to the feature map space
        self.ndf = ndf
        self.out_size = im_size // 32
        self.decoder_dense = nn.Sequential(
            nn.Linear(z_size, ndf * 16 * self.out_size * self.out_size) #, nn.ReLU()
        )
        # Decoder: (ndf*8, im_size//16, im_size//16) -> (num_ch, im_size, im_size)
        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ndf*16, ndf*8, 3, padding=1),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ndf*8, ndf*4, 3, padding=1),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ndf*4, ndf*2, 3, padding=1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ndf*2, ndf, 3, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2),


            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ndf, ndf, 3, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(),
            nn.Conv2d(ndf, num_ch, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        batch_size = input.size(0)
        hidden = self.decoder_dense(input).view(
            batch_size, self.ndf*16, self.out_size, self.out_size)
        output = self.decoder_conv(hidden)
        return output


class VAE(nn.Module):
    def __init__(self, num_ch=3, ndf=32, nef=32, z_size=100, im_size=64, device=torch.device("cuda:0"), is_train=True):
        super(VAE, self).__init__()

        self.z_size = z_size
        self.im_size=im_size
        # Encoder
        self.encoder = Encoder(num_ch=num_ch, nef=nef, z_size=z_size, im_size=im_size, device=device)
        # Decoder
        self.decoder = Decoder(num_ch=num_ch, ndf=ndf, z_size=z_size, im_size=im_size)

        if is_train == False:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        latent_z, mean, logvar = self.encoder(x)
        rec_x = self.decoder(latent_z)
        return rec_x, mean, logvar
    
    def encode(self, x):
        latent_z, _, _ = self.encoder(x)
        return latent_z

    def decode(self, z):
        return self.decoder(z)

    def sample(self, size):
        sample = torch.randn(size, self.z_size).to(self.device)
        return model.decode(sample)
    
    @property
    def device(self): return next(self.parameters()).device