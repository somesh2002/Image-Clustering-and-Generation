import sys

#Any additional sklearn import will flagged as error in autograder
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from scipy.stats import multivariate_normal
import pickle
import pandas as pd



torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class SubsetMNIST(Dataset):
    def __init__(self, npz_file_path):
        self.data = np.load(npz_file_path)
        self.features = self.data['data']
        self.labels = self.data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(1)
        feature = feature / 255.0
        label = torch.tensor(label, dtype=torch.long)
        return feature, label

class Encoder(nn.Module):
    def __init__(self, image_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(image_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, image_dim):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, image_dim)
    
    def forward(self, z):
        h = torch.relu(self.fc3(z))
        h = torch.relu(self.fc4(h))
        final_image = torch.sigmoid(self.fc5(h))
        return final_image

# VAE Model
class VAE(nn.Module):
    def __init__(self, image_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(image_dim, latent_dim)
        self.decoder = Decoder(latent_dim, image_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x.view(-1, image_dim))
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta_value):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, image_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    beta = beta_value
    return BCE + beta*KLD

def extract_latent_vectors(data_loader, model):
    model.eval()
    latent_vectors = []
    labels = []
    with torch.no_grad():
        for images, label in data_loader:
            images = images.to(device)
            mu, _ = model.encoder(images.view(-1, 28 * 28))  # Extract the mean (mu)
            latent_vectors.append(mu)
            labels.append(label)
    latent_vectors = torch.cat(latent_vectors, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    return latent_vectors, labels

class GMM:
    def __init__(self, clusters):
        self.clusters = clusters  # Number of clusters
        self.means = []
        self.covariances =[]
        self.weights= []
 
    def initialize_parameters(self, data,labels):
        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_features = data[labels == label]
            average_features = 2*label_features.mean(axis=0)
            self.means.append(average_features)
        self.covariances = [np.eye(data.shape[1])/30 for _ in range(self.clusters)]
        self.weights = np.ones(self.clusters) / self.clusters
    
    def expectation_step(self,data):
        n_samples = data.shape[0]
        cond_prob = np.zeros((n_samples, self.clusters))

        for k in range(self.clusters):
            pdf = multivariate_normal.pdf(data, mean=self.means[k], cov=self.covariances[k])
            cond_prob[:, k] = self.weights[k] * pdf
        cond_prob /= cond_prob.sum(axis=1, keepdims=True)
        return cond_prob
    
    def maximization_step(self,train_data, cond_prob):
        n_samples, n_features = train_data.shape
        total_component = self.clusters

        weights = np.zeros(total_component)
        means = np.zeros((total_component, n_features))
        covariances = np.zeros((total_component, n_features, n_features))

        for k in range(total_component):
            total_likelihood = cond_prob[:, k].sum()
            weights[k] = total_likelihood / n_samples
            means[k] = (train_data * cond_prob[:, k][:, np.newaxis]).sum(axis=0) / total_likelihood
            diff = train_data - means[k]
            covariances[k] = (cond_prob[:, k][:, np.newaxis, np.newaxis] *
                            (diff[:, :, np.newaxis] * diff[:, np.newaxis, :])).sum(axis=0) / total_likelihood
        self.weights = weights
        self.means = means
        self.covariances = covariances
        return
    
    def assign_labels(self,train_data,train_labels):
        cond_prob = self.expectation_step(train_data)
        components = self.clusters
        self.labels = np.zeros(components, dtype=int)
        for k in range(components):
            indices = np.argmax(cond_prob, axis=1) == k
            component_labels = train_labels[indices]
            if component_labels.size > 0:
                most_common_label = Counter(component_labels).most_common(1)[0][0]
                self.labels[k] = most_common_label

    def predict(self, X):
        cond_prob = self.expectation_step(X)
        return np.argmax(cond_prob, axis=1)
    
    def save_parameters(self, filename):
        save_weights = {
            'means': self.means,
            'covariances': self.covariances,
            'weights': self.weights,
            'labels': self.labels
        }
        with open(filename, 'wb') as f:
            pickle.dump(save_weights, f)
    
    def load_parameters(self, filename):
        with open(filename, 'rb') as f:
            params = pickle.load(f)
        self.means = params['means']
        self.covariances = params['covariances']
        self.weights = params['weights']
        self.labels = params['labels']
    



if __name__ == "__main__": 
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    arg3 = sys.argv[3] if len(sys.argv) > 3 else None
    arg4 = sys.argv[4] if len(sys.argv) > 4 else None
    arg5 = sys.argv[5] if len(sys.argv) > 5 else None

if len(sys.argv)==4:### Running code for vae reconstruction.
    path_to_test_dataset_recon = arg1
    test_reconstruction = arg2
    vaePath = arg3

    test_dataset = SubsetMNIST(path_to_test_dataset_recon)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    image_dim = 28 * 28
    latent_dim = 2
    model = VAE(image_dim, latent_dim).to(device)
    model.load_state_dict(torch.load(vaePath))
    model.eval()
    test_images = []
    for images, lbls in test_loader:    
        images = images.to(device)
        recon_data, _, _ = model(images)
        new_image = recon_data.cpu().view(28, 28).detach().numpy()
        test_images.append(new_image)
    test_array = np.array(test_images)
    np.savez("vae_reconstructed.npz", data = test_images)
    
elif len(sys.argv)==5:###Running code for class prediction during testing
    path_to_test_dataset = arg1
    test_classifier = arg2
    vaePath = arg3
    gmmPath = arg4


    test_dataset = SubsetMNIST(path_to_test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    

    image_dim = 28 * 28
    latent_dim = 2
    model = VAE(image_dim, latent_dim).to(device)
    model.load_state_dict(torch.load(vaePath))
    model.eval()

    n_clusters = 3
    gmm = GMM(n_clusters)
    gmm.load_parameters(gmmPath)
    test_labels = []
    for images, lbls in test_loader:    
        images = images.to(device)
        latent_vector, _  = model.encoder(images.view(-1,image_dim))
        latent_vector   = latent_vector.cpu().detach().numpy()    
        final_label = gmm.predict(latent_vector)
        class_assigned = gmm.labels[final_label]
        test_labels.extend(class_assigned)

    labels_df = pd.DataFrame(test_labels, columns=["Predicted_Label"])
    labels_df.to_csv("vae.csv", index=False)

else:### Running code for training. save the model in the same directory with name "vae.pth"
    path_to_train_dataset = arg1
    path_to_val_dataset = arg2
    trainStatus = arg3
    vaePath = arg4
    gmmPath = arg5
    torch.manual_seed(0)

    train_batch_size = 64
    train_dataset = SubsetMNIST(path_to_train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    val_batch_size = 64
    val_dataset = SubsetMNIST(path_to_val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    image_dim = 28 * 28
    latent_dim = 2
    vae = VAE(image_dim, latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=9e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 20, gamma= 0.5)
    beta_value = 1

    epochs = 100
    for epoch in range(epochs):
        vae.train()
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch[0]
            x = x.to(device)
            recon_x, mu, logvar = vae(x)
            loss = vae_loss(recon_x, x, mu, logvar, beta_value)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
    torch.save(vae.state_dict(), vaePath)

    train_latent_vectors, train_labels = extract_latent_vectors(train_loader, vae)
    val_latent_vectors, val_labels = extract_latent_vectors(val_loader,vae)

    n_clusters = 3
    gmm = GMM(n_clusters)
    gmm.initialize_parameters(val_latent_vectors,val_labels)
    max_iter = 20
    for _ in range(max_iter):
        probs = gmm.expectation_step(train_latent_vectors)
        gmm.maximization_step(train_latent_vectors, probs)
    gmm.assign_labels(val_latent_vectors,val_labels)
    gmm.save_parameters(gmmPath)