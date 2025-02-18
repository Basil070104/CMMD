import torch
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
from mmd import MMD
import matplotlib.pyplot as plt

class CLIP:
  
  def __init__(self):
    """Initialize CLIP model and processor with the correct variant"""
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
    self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")  # Fixed processor
    
  def normalize(self, normalize):
    """Normalize"""
    z_list = list()
    for score in normalize:
      z = (score - min(normalize)) / (1 - min(normalize))
      z_list.append(z)
      
    return z_list
    
  
  def image_and_text_feature(self):
    """Extracts CLIP features from text and images and computes MMD"""
    
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_inputs = tokenizer(["a parrot"], padding=True, return_tensors="pt").to(self.device)
    text_features = self.model.get_text_features(**text_inputs)

    # url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/Pantheon_Rom_1_cropped.jpg/1920px-Pantheon_Rom_1_cropped.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    # image = Image.open("/home/bkhwaja/vscode/ECE50024/images/distorted.png")
    # image2 = Image.open("/home/bkhwaja/vscode/ECE50024/images/clean.png")
    # image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
    # image_inputs_2 = self.processor(images=image2, return_tensors="pt").to(self.device)
    # image_features_1 = self.model.get_image_features(**image_inputs)
    # image_features_2 = self.model.get_image_features(**image_inputs_2)
    # text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    # image_features_1 = image_features_1 / image_features_1.norm(p=2, dim=-1, keepdim=True)
    # image_features_2 = image_features_2 / image_features_2.norm(p=2, dim=-1, keepdim=True)
    # text_distribution = text_features.cpu().detach().numpy()
    # image_distribution_1 = image_features_1.cpu().detach().numpy()
    # image_distribution_2 = image_features_2.cpu().detach().numpy()
    # mmd = MMD(X=image_distribution_1.T, Y=image_distribution_2.T, sigma=1000, scale=100)
    # comparison = mmd.compute_mmd()  # No multiplication with logit_scale
    # print(f"The comparison between the text and image is: {comparison}")
    # return text_distribution, image_distribution_2
    comparison_list = list()
    # comparison_list.append(0)
    for i in range(2, 5):
      image = Image.open("./images/distorted" + str(i) + ".png")
      image2 = Image.open("./images/cleanParrot.png")

      image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
      image_inputs_2 = self.processor(images=image2, return_tensors="pt").to(self.device)
      image_features_1 = self.model.get_image_features(**image_inputs)
      image_features_2 = self.model.get_image_features(**image_inputs_2)
      
      text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
      text_distribution = text_features.cpu().detach().numpy()
      
      image_distribution_1 = image_features_1.cpu().detach().numpy()
      image_distribution_2 = image_features_2.cpu().detach().numpy()

      mmd = MMD(X = image_distribution_1.T, Y = image_distribution_2.T, sigma = 0.5, scale=100)
      comparison = mmd.compute_mmd()
      # print(f"The comparison between the distorted image id {i} and image is: {comparison}")
      
      comparison_list.append(100 * comparison)
    
    # z_list = self.normalize(comparison_list)
    # for z in z_list:
    #   print(f"The normalized comparison to the original image : {1000 * z}")
    print(comparison_list)
    image_paths = [
      # "./images/cleanParrot.png",
      "./images/distorted2.png",
      "./images/distorted3.png",
      "./images/distorted4.png"
    ]

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    axs = axs.flatten()
    for ax, img_path, z in zip(axs, image_paths, comparison_list):
      img = Image.open(img_path) 
      ax.imshow(img) 
      ax.axis('off') 
      ax.text(0.5, 0.95, f'Z-score: {z:.4f}', fontsize=12, ha='center', va='center', color='white', 
            bbox=dict(facecolor='black', alpha=1.0, edgecolor='none'))
      
    plt.tight_layout()
    plt.show()
  
def main():
  clip = CLIP()
  clip.image_and_text_feature()  

if __name__ == "__main__":
  main()