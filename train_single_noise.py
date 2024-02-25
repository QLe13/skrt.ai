# Import of libraries
import random
import imageio
import numpy as np
from argparse import ArgumentParser

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST
import string   
import torchvision.utils as vutils
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import time
# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Definitions
STORE_PATH_MNIST = f"ddpm_model_mnist.pt"
STORE_PATH_FASHION = f"ddpm_model_fashion.pt"

no_train = False
fashion = True

n_epochs = 20
lr = 0.001

store_path = "ddpm_fashion.pt" if fashion else "ddpm_mnist.pt"

def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()

def show_first_batch(loader):
    for batch in loader:
        show_images(batch[0], "Images in the first batch")
        break

authors = ["Author" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)) for _ in range(20)]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def create_citation(label, label_count):
    class_name = class_names[label]
    author_name = f"Author_{label_count}"
    citation = f"{class_name}_{author_name}"
    return citation

def tokenize_citation(citation):
    tokens = []
    print(citation)
    for part in citation.split('_'):
        if part[0].isdigit():
            tokens.extend(list(part))  # Add each digit as a separate token
        else:
            tokens.append(part)
    return tokens

import torch
from torch.utils.data import Dataset

class CitationDataset(Dataset):
    def __init__(self, dataset, class_names, authors, token_to_id, citation_embedding, sequence_length=14, save_path=None, load_precomputed=False):
        self.dataset = dataset
        self.label_counts = {label: 0 for label in range(len(class_names))}
        self.token_to_id = token_to_id
        self.sequence_length = sequence_length
        self.precomputed_noise = []
        self.citations = []
        self.device = device
        self.save_path = save_path

        if load_precomputed and save_path:
            self.load_precomputed_data()
        else:
            self.precompute_data(citation_embedding)
            if save_path:
                self.save_precomputed_data()

    def precompute_data(self, citation_embedding):
        for i,(image, label) in enumerate(self.dataset):
            if i %  1000 == 0:
                print("i", label)
            num = self.label_counts[label]
            self.label_counts[label] += 1
            citation = create_citation(label, num)
            self.citations.append(citation)
            citation = self._citation_to_tensor(citation).to(device)
            citation = citation.unsqueeze(0)
            citation_tensor = citation_embedding(citation)
            self.precomputed_noise.append(citation_tensor)

    def save_precomputed_data(self):
        torch.save({
            'precomputed_noise': self.precomputed_noise,
            'citations': self.citations
        }, self.save_path)
        print(f"Saved precomputed data to {self.save_path}")

    def load_precomputed_data(self):
        data = torch.load(self.save_path)
        self.precomputed_noise = data['precomputed_noise']
        self.citations = data['citations']
        print(f"Loaded precomputed data from {self.save_path}")

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        precomp_noise = self.precomputed_noise[idx]
        citation = self.citations[idx]

        return image, label, precomp_noise, citation

    def __len__(self):
        return len(self.dataset)

    def _citation_to_tensor(self, citation):
        tokens = tokenize_citation(citation)
        token_ids = [self.token_to_id[token] for token in tokens if token in self.token_to_id]

        token_ids += [self.token_to_id["<PAD>"]] * (self.sequence_length - len(token_ids))
        citation_tensor = torch.tensor(token_ids, dtype=torch.long)
        return citation_tensor

def _citation_to_tensor_batch(citations):
    sequence_length = 14
    
    batch_token_ids = []
    for citation in citations:
        tokens = tokenize_citation(citation)
        token_ids = [token_to_id[token] for token in tokens if token in token_to_id]

        # Padding
        token_ids += [token_to_id["<PAD>"]] * (sequence_length - len(token_ids))
        batch_token_ids.append(token_ids)

    # Convert the list of token_ids into a tensor
    citation_tensor_batch = torch.tensor(batch_token_ids, dtype=torch.long)
    return citation_tensor_batch

# Getting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))

class Encoder(nn.Module):
    def __init__(self, vocab_size, nhead, nhid, nlayers):
        super(Encoder, self).__init__()
        self.sequence_length = 14  # Fixed based on input data
        self.embed_dim = 56  # Set to ensure 8 * embed_dim is a perfect square <= 256

        self.embedding = nn.Embedding(vocab_size, self.embed_dim)
        encoder_layers = TransformerEncoderLayer(self.embed_dim, nhead, nhid)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        # src: [batch_size, sequence_length]
        embedded = self.embedding(src)  # [batch_size, sequence_length, embed_dim]
        embedded = embedded.permute(1, 0, 2)  # [sequence_length, batch_size, embed_dim]
        output = self.transformer_encoder(embedded)  # [sequence_length, batch_size, embed_dim]

        # Reshape to [batch_size, x, x], where x*x = sequence_length * embed_dim
        output = output.permute(1, 0, 2)  # [batch_size, sequence_length, embed_dim]
        output = output.reshape(-1, 28, 28)  # [batch_size, 16, 16]
        return output

class Decoder(nn.Module):
    def __init__(self, vocab_size, nhead, nhid, nlayers):
        super(Decoder, self).__init__()
        self.embed_dim = 56  # Keep consistent with Encoder

        self.embedding = nn.Embedding(vocab_size, self.embed_dim)
        decoder_layers = TransformerDecoderLayer(self.embed_dim, nhead, nhid)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.output_layer = nn.Linear(self.embed_dim, vocab_size)

    def forward(self, src, memory):
        # src: [batch_size, sequence_length]
        # memory: [batch_size, 28, 28], need to reshape correctly for TransformerDecoder
        
        # Correctly reshape and permute memory for multi-head attention
        # Flatten the spatial dimensions and keep the batch_size dimension intact
        memory_reshaped = memory.view(memory.size(0), -1, self.embed_dim)  # Reshape to [batch_size, 784, embed_dim]
        memory_reshaped = memory_reshaped.permute(1, 0, 2)  # Permute to [sequence_length, batch_size, embed_dim]

        tgt_embedded = self.embedding(src).permute(1, 0, 2)  # [sequence_length, batch_size, embed_dim]

        output = self.transformer_decoder(tgt_embedded, memory_reshaped)
        output = output.permute(1, 0, 2)  # [batch_size, sequence_length, embed_dim]

        logits = self.output_layer(output)
        return logits

# DDPM class
class MyDDPM(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None, image_chw=(1, 28, 28), max_citations=500):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
            device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)
        self.max_citations = max_citations

    def forward(self, x0, t, precomputed_citation):
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if precomputed_citation is None:
            raise ValueError("Precomputed citations must be provided")

        # Apply the noise using precomputed citations instead of random noise
        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * precomputed_citation
        #noise = noisy - x0
        return noisy #, noise

    def backward(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        noise = self.network(x, t)
        reversed_images = x - noise
        return noise, reversed_images

def compare_encs(ddpm, citation_embedding, loader, device):
    # Showing the forward process
    for batch in loader:
        with torch.no_grad():
            if device is None:
                device = ddpm.device
            x0, _, citations, ct = batch  # Assuming citations are the second element now
            citations = citations.to(device)
            embedding = citation_embedding(citations)
            embedding = (embedding - embedding.min()) / (embedding.max() - embedding.min()) * 2 - 1 #SCALING MAY NEED FIX
            # Initialize x with a single image prior

            x = [get_img_prior(ddpm, device, ddpm.n_steps, embedding=i) for i in embedding]
            x = torch.stack(x, dim=0)
            print("x shape", x.shape)
            show_images(x, "Prior Based image")

            t = [ddpm.n_steps - 1 for _ in range(len(x0))]
            print("x0", x0.shape)
            print("embedding", embedding.shape)
            # Apply forward diffusion process
            x = ddpm(x0, t, embedding)
            show_images(x, "Original DS image")
        return

def show_forward(ddpm, loader, device):
    # Showing the forward process
    for batch in loader:
        imgs, _, noise, ct = batch  # Assuming citations are the second element now
        show_images(imgs, "Original images")
        noise = noise.to(device)
        #citations = citations.to(device)
        #embedding = citation_embedding(citations)
        #embedding = (embedding - embedding.min()) / (embedding.max() - embedding.min()) * 2 - 1
        #show_frequency_domain(imgs, "Original images")

        for percent in [0.25, 0.5, 0.75, 1]:
            noised_imgs = ddpm(imgs.to(device),
                               [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))],
                               noise)
            #noised_imgs = noised_imgs.squeeze(2)
            #print(noised_imgs.shape)
            show_images(noised_imgs, f"DDPM Noisy images {int(percent * 100)}%")
            #show_frequency_domain(noised_imgs, f"DDPM Noisy images {int(percent * 100)}%")
            #show_citation_bars(citations[:num_images], f"Citations at {int(percent * 100)}% Noise")
        break  # Only show for the first batch

def show_images(images, title):
    # Check if the input is a single image (2D tensor), and add a batch dimension if necessary
    if len(images.shape) == 2:
        images = images.unsqueeze(0)  # Add batch and channel dimensions

    print("Displaying images with shape:", images.shape)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    # Convert to grid and transpose to make it compatible with plt.imshow
    plt.imshow(np.transpose(vutils.make_grid(images, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

def show_frequency_domain(images, title):
    """Display the frequency domain representation of each image in a batch."""
    n_images = images.shape[0]
    grid_size = int(np.ceil(np.sqrt(n_images)))

    plt.figure(figsize=(10, 10))
    plt.suptitle(title)

    for i in range(n_images):
        # Extract the image and convert to frequency domain
        img = images[i].detach().cpu().squeeze()  # Ensure it's a single-channel 2D tensor
        freq_domain = torch.fft.fft2(img)
        freq_domain = torch.fft.fftshift(freq_domain)  # Shift zero frequency component to the center

        # Calculate magnitude spectrum (log scale for better visibility)
        magnitude = torch.log(torch.abs(freq_domain) + 1e-10)

        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(magnitude.numpy(), cmap='gray')
        plt.axis('off')

    plt.show()


def get_img_prior(ddpm, device, n_steps, embedding):

    # Create a white image of size (1, 28, 28) with correct dimensions
    #white_image = torch.ones((1, 1, 28, 28))
    white_image = torch.full((1, 1, 28, 28), -1)
    white_image = white_image.to(torch.float32)
    #white_image = torch.zeros((1, 1, 28, 28))
    # Apply forward diffusion process on the white image at different noise levels
    # Create a tensor of time steps for the entire batch
    t = torch.full((1,), n_steps - 1, dtype=torch.long)

    # Process the white image through the DDPM
    noised_img = ddpm(white_image.to(device), t.to(device), embedding.to(device))
    
    return noised_img.squeeze(0)  # Remove the extra batch dimension

def decode_sequence(logits, id_to_token):
    # Convert logits to predicted class indices
    # logits shape: [batch_size, sequence_length, vocab_size]
    predictions = torch.argmax(logits, dim=-1)

    # Convert predicted indices to tokens
    decoded_sequences = []
    for sequence in predictions:
        tokens = [id_to_token[idx.item()] for idx in sequence]
        decoded_sequences.append(tokens)

    def lists_to_strings(lists_of_strings):
        def process_list(lst):
            # Filter out '<PAD>' tokens
            filtered_lst = [item for item in lst if item != '<PAD>']
            
            # Initialize containers for processed parts
            non_numeric_parts = []
            numeric_part = ''
            
            for item in filtered_lst:
                # Check if item is numeric
                if item.isdigit():
                    numeric_part += item  # Concatenate numeric strings
                else:
                    # If there's a numeric part already formed and we encounter a non-numeric item,
                    # add the numeric part to non_numeric_parts and reset numeric_part
                    if numeric_part:
                        non_numeric_parts.append(numeric_part)
                        numeric_part = ''
                    non_numeric_parts.append(item)  # Add non-numeric item
            
            # Add any remaining numeric part to the list
            if numeric_part:
                non_numeric_parts.append(numeric_part)
            
            # Join all parts with an underscore
            return "_".join(non_numeric_parts)
    
        # Process each list in the input list of lists
        return [process_list(lst) for lst in lists_of_strings]

    return lists_to_strings(decoded_sequences)

def generate_new_images_adapt(ddpm, citation_embedding, citation_decoder, device=None, frames_per_gif=100, gif_name="sampling.gif", loader=None):
    #frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    #frames = []
    for batch in loader:
        with torch.no_grad():
            if device is None:
                device = ddpm.device
            _, _, citations, ct = batch  # Assuming citations are the second element now
            #print("Citation: ", ct)
            citations = citations.to(device)
            #embedding = citation_embedding(citations)
            #embedding = (embedding - embedding.min()) / (embedding.max() - embedding.min()) * 2 - 1 #SCALING MAY NEED FIX
            # Initialize x with a single image prior

            x = [get_img_prior(ddpm, device, ddpm.n_steps, embedding=i) for i in citations]
            x = torch.stack(x, dim=0)
            print("x shape", x.shape)
            show_images(x, "Original image")
            t = torch.full((batch_size,), ddpm.n_steps - 1, dtype=torch.long).to(device)

            ns, recons = ddpm.backward(x, t)
            #print(ns.shape)
            show_images(recons, "Reconstructed image 1")
            citation = _citation_to_tensor_batch(ct).to(device)
            rec_cite = citation_decoder(citation, ns)
            decoded_sequences = decode_sequence(rec_cite, id_to_token)
            # Print decoded sequences
            for i in range(len(decoded_sequences)):
                print("Original: ", ct[i])
                print("Reconstructed: ", decoded_sequences[i])
        return recons.squeeze()  # Remove batch dimension for the final image

def generate_new_images_adapt_ds(ddpm, citation_embedding, device=None, frames_per_gif=100, gif_name="sampling.gif", loader=None):
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []

    for batch in loader:
        with torch.no_grad():
            if device is None:
                device = ddpm.device

            # Initialize x with a single image prior
            x0, _, citations, ct = batch
            show_images(x0, "Original images")
            print("Citation: ", ct)
            #print(citation_dataset[3][3])
            #print("Citation: ", ct)
            #x0 = x0.unsqueeze(0)
            x0 = x0.to(device)
            citations = citations.to(device)
            #embedding = citation_embedding(citations)
            #embedding = (embedding - embedding.min()) / (embedding.max() - embedding.min()) * 2 - 1 #SCALING MAY NEED FIX
            t = [ddpm.n_steps - 1 for _ in range(len(x0))]
            mse = nn.MSELoss()
            #t = torch.tensor([n_steps]).to(device) #torch.full((1,), ddpm.n_steps - 1, dtype=torch.long)
            #print("t", t.shape)
            # Apply forward diffusion process
            x = ddpm(x0, t, citations)
            show_images(x, "Original DS image")
            t = torch.full((batch_size,), ddpm.n_steps - 1, dtype=torch.long).to(device)
            #torch.tensor(t).to(device).long()
            ns, recons = ddpm.backward(x, t)
            print("MSE", mse(x0, recons))
            show_images(recons, "Reconstructed images")
            citation = _citation_to_tensor_batch(ct).to(device)
            rec_cite = citation_decoder(citation, ns)
            decoded_sequences = decode_sequence(rec_cite, id_to_token)
            # Print decoded sequences
            for i in range(len(decoded_sequences)):
                print("Original: ", ct[i])
                print("Reconstructed: ", decoded_sequences[i])
        return recons.squeeze()  # Remove batch dimension for the final image

def sinusoidal_embedding(n, d):
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])
    return embedding

class ResidualBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(ResidualBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding, bias=False)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize
        self.residual = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        out += identity  # Adding the skip connection
        return out

class MyUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(MyUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            ResidualBlock((1, 28, 28), 1, 16),
            ResidualBlock((16, 28, 28), 16, 16),
            ResidualBlock((16, 28, 28), 16, 16)
        )
        self.down1 = nn.Conv2d(16, 16, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 16)
        self.b2 = nn.Sequential(
            ResidualBlock((16, 14, 14), 16, 32),
            ResidualBlock((32, 14, 14), 32, 32),
            ResidualBlock((32, 14, 14), 32, 32)
        )
        self.down2 = nn.Conv2d(32, 32, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 32)
        self.b3 = nn.Sequential(
            ResidualBlock((32, 7, 7), 32, 64),
            ResidualBlock((64, 7, 7), 64, 64),
            ResidualBlock((64, 7, 7), 64, 64)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 64, 2, 1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 64)
        self.b_mid = nn.Sequential(
            ResidualBlock((64, 3, 3), 64, 32),
            ResidualBlock((32, 3, 3), 32, 32),
            ResidualBlock((32, 3, 3), 32, 64)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 64, 2, 1)
        )

        self.te4 = self._make_te(time_emb_dim, 128)
        self.b4 = nn.Sequential(
            ResidualBlock((128, 7, 7), 128, 64),
            ResidualBlock((64, 7, 7), 64, 32),
            ResidualBlock((32, 7, 7), 32, 32)
        )

        self.up2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 64)
        self.b5 = nn.Sequential(
            ResidualBlock((64, 14, 14), 64, 32),
            ResidualBlock((32, 14, 14), 32, 16),
            ResidualBlock((16, 14, 14), 16, 16)
        )

        self.up3 = nn.ConvTranspose2d(16, 16, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 32)
        self.b_out = nn.Sequential(
            ResidualBlock((32, 28, 28), 32, 16),
            ResidualBlock((16, 28, 28), 16, 16),
            ResidualBlock((16, 28, 28), 16, 16, normalize=False)
        )

        self.conv_out = nn.Conv2d(16, 1, 3, 1, 1)

    def forward(self, x, t):
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1)) 
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1) 
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))

        out5 = torch.cat((out2, self.up2(out4)), dim=1) 
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))

        out = torch.cat((out1, self.up3(out5)), dim=1) 
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))

        out = self.conv_out(out)
        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

no_train = False
fashion = True
batch_size = 128
n_epochs = 20
lr = 0.001
nhead = 8  
nhid = 256      # Dimension of the feedforward network model in transformer layers (4 times embed_dim)
nlayers = 3  
# Example usage
pad_token = "<PAD>"
digits = [str(i) for i in range(10)]
unique_tokens = set(class_names + ["Author"] + digits + ["_", pad_token])

token_to_id = {token: id for id, token in enumerate(unique_tokens)}
id_to_token = {id: token for token, id in token_to_id.items()}

#REPEAT
unique_tokens = set(class_names + authors + ["_"])
vocab_size = len(unique_tokens)

transform = Compose([
    ToTensor(),
    Lambda(lambda x: (x - 0.5) * 2)]
)

citation_embedding = Encoder(vocab_size=vocab_size, nhead=nhead, nhid=nhid, nlayers=nlayers)
citation_embedding.load_state_dict(torch.load("mono_enc.pt"))
citation_embedding = citation_embedding.to(device)
inverse_citation_embedding = Decoder(vocab_size=vocab_size, nhead=nhead, nhid=nhid, nlayers=nlayers)
inverse_citation_embedding.load_state_dict(torch.load("mono_dec.pt"))
inverse_citation_embedding = inverse_citation_embedding.to(device)

for param in citation_embedding.parameters():
    param.requires_grad = False

# Freeze all the weights in modelA
for param in inverse_citation_embedding.parameters():
    param.requires_grad = False

#ds_fn = FashionMNIST if fashion else MNIST
#dataset = ds_fn("./datasets", download=True, train=True, transform=transform)
#loader = DataLoader(dataset, batch_size, shuffle=True)


#loader = DataLoader(dataset, batch_size, shuffle=True)


ds_fn = FashionMNIST if fashion else MNIST
dataset = ds_fn("./datasets", download=True, train=True, transform=transform)
citation_dataset = CitationDataset(dataset, class_names, authors, token_to_id, citation_embedding, save_path='28x28_noise.pth', load_precomputed=True)

from torch.utils.data import random_split

# Split dataset into training and validation sets
total_size = len(citation_dataset)
val_size = int(total_size * 0.2)  # 20% for validation
train_size = total_size - val_size

train_dataset, val_dataset = random_split(citation_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)  # No need to shuffle validation data

# Defining model
n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors
ddpm = MyDDPM(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)
#ddpm.load_state_dict(torch.load(store_path, map_location=device))
ddpm = ddpm.to(device)
sum([p.numel() for p in ddpm.parameters()])
#get_img_prior(ddpm, device, n_steps, "Ankle boot_Author_0")
# Optionally, load a pre-trained model that will be further trained
# ddpm.load_state_dict(torch.load(store_path, map_location=device))

# Optionally, show the diffusion (forward) process
show_forward(ddpm, train_loader, device)
#compare_encs(ddpm, citation_embedding, train_loader, device)
# Optionally, show the denoising (backward) process
#generated = generate_new_images_adapt(ddpm, gif_name="before_training.gif")
#show_images(generated, "Images generated before training")

def training_loop(ddpm, train_loader, val_loader, n_epochs, optim, device, display=False, store_path="ddpm_model.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps
    lambda_citation = 0.1 
    lambda_prior = 0.01

    for epoch in tqdm(range(n_epochs), desc="Training progress", colour="#87CEEB"):
        train_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#87CEEB")):
            # Loading data
            x0, _, precomp_noise, ct = batch
            
            x0 = x0.to(device)
            #citations = citations.to(device)
            precomp_noise = precomp_noise.to(device)
            n = x0.size(0)

            t = torch.randint(0, n_steps, (n,)).to(device)
            noisy_imgs = ddpm.forward(x0, t, precomp_noise)
            #show_images(noisy_imgs, f"Noisy Imgs {epoch + 1}")
            # Getting model estimation of noise based on the images and the time-step
            estimated_noise, rec_imgs = ddpm.backward(noisy_imgs, t)

            #loss = mse(estimated_noise, precomp_noise) + mse(rec_imgs, x0)
            loss = mse(rec_imgs, x0)
            train_loss += loss.item()
            """
            if loss < 0.01:
                print(ct)
                show_images(reconstructed_imgs, f"Images generated at epoch {epoch + 1}")
                show_images(x0, f"Original images at epoch {epoch + 1}")
            """
            optim.zero_grad()
            loss.backward()
            optim.step()

        citation_embedding.eval()
        ddpm.eval()
        val_loss = 0.0
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                x0, _, precomp_noise, ct = batch
            
                x0 = x0.to(device)
                #citations = citations.to(device)
                precomp_noise = precomp_noise.to(device)
                n = x0.size(0)

                t = torch.randint(0, n_steps, (n,)).to(device)
                noisy_imgs = ddpm.forward(x0, t, precomp_noise)
                #show_images(noisy_imgs, f"Noisy Imgs {epoch + 1}")
                # Getting model estimation of noise based on the images and the time-step
                estimated_noise, rec_imgs = ddpm.backward(noisy_imgs, t)

                #loss = mse(estimated_noise, precomp_noise) #+ mse(rec_imgs, x0)
                loss = mse(rec_imgs, x0)
                val_loss += loss.item()

        # Print average losses for the epoch
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{n_epochs}, Average Training Loss: {avg_train_loss}, Average Validation Loss: {avg_val_loss}")
        torch.save(ddpm.state_dict(), "ddpm_mono_28_orig.pt")

"""       
# Training
store_path = "ddpm_fashion.pt" if fashion else "ddpm_mnist.pt"
if not no_train:
    training_loop(ddpm, train_loader, val_loader, n_epochs, optim=Adam(ddpm.parameters(), lr), device=device, store_path=store_path)
"""

# Loading the trained model
best_model = MyDDPM(MyUNet(), n_steps=n_steps, device=device)
weight_path = "ddpm_mono_28.pt"
best_model.load_state_dict(torch.load(weight_path, map_location=device))
best_model.eval()
best_model.to(device)
print("Model loaded")

print("Generating new images")
generated = generate_new_images_adapt(
        best_model,
        citation_embedding=citation_embedding,
        citation_decoder=inverse_citation_embedding,
        device=device,
        gif_name="fashion.gif" if fashion else "mnist.gif",
        loader=val_loader
    )
#show_images(generated, "Final result")

generated_ds = generate_new_images_adapt_ds(
        best_model,
        citation_embedding=citation_embedding,
        citation_decoder=inverse_citation_embedding,
        device=device,
        gif_name="fashion_ds.gif" if fashion else "mnist_ds.gif",
        loader=val_loader
    )
#show_images(generated_ds, "Final result DS")
