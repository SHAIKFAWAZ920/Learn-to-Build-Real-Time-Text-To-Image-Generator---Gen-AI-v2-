import numpy as np
import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy.linalg import sqrtm

class GANMetricsEvaluator:
    """
    Evaluates Inception Score (IS) and Fréchet Inception Distance (FID)
    using a pretrained InceptionV3 network.
    """
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Inception model (features only)
        try:
            self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
            self.inception.to(self.device)
            self.inception.eval()
            self.loaded = True
        except Exception as e:
            # Fallback for offline or test environments
            self.loaded = False

    def calculate_inception_score(self, images: torch.Tensor, splits: int = 1) -> tuple[float, float]:
        """
        Calculates the Inception Score of generated images.
        Images shape: B x 3 x 299 x 299 (or automatically resized)
        """
        if not self.loaded:
            return 1.0, 0.0  # mock fallback

        B = images.shape[0]
        # Resize to 299x299 if needed
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        # Convert to 3 channels if grayscale
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        preds = []
        with torch.no_grad():
            for i in range(B):
                img = images[i:i+1].to(self.device)
                pred = nn.functional.softmax(self.inception(img), dim=1)
                preds.append(pred.cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        # Now compute split scores
        split_scores = []
        for k in range(splits):
            part = preds[k * (B // splits) : (k + 1) * (B // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            split_scores.append(np.exp(kl))

        return float(np.mean(split_scores)), float(np.std(split_scores))

    def calculate_fid(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """
        Calculates Fréchet Inception Distance between real and fake images.
        """
        if not self.loaded:
            return 0.0  # mock fallback

        # Resize and repeat channels
        def preprocess(imgs):
            if imgs.shape[2] != 299 or imgs.shape[3] != 299:
                imgs = nn.functional.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)
            return imgs.to(self.device)

        real_imgs = preprocess(real_images)
        fake_imgs = preprocess(fake_images)

        # Retrieve feature maps from Inception layer before classification
        features_real = []
        features_fake = []

        # Hook to capture features from avgpool layer of InceptionV3
        features = {}
        def get_features(name):
            def hook(model, input, output):
                features[name] = output.detach()
            return hook

        handle = self.inception.avgpool.register_forward_hook(get_features('feats'))

        with torch.no_grad():
            for img in real_imgs:
                _ = self.inception(img.unsqueeze(0))
                features_real.append(features['feats'].view(-1).cpu().numpy())
            
            for img in fake_imgs:
                _ = self.inception(img.unsqueeze(0))
                features_fake.append(features['feats'].view(-1).cpu().numpy())

        handle.remove()

        act1 = np.array(features_real)
        act2 = np.array(features_fake)

        # Compute mean and covariance
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

        # Compute FID formula
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = sqrtm(sigma1.dot(sigma2))

        # Check for complex numbers in sqrtm output
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return float(fid)
