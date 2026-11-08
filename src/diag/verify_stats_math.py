import numpy as np

data = np.load("data/checkpoints_siglip-vit/vit_zero_train_stats.npz")
mu_img = data["mu_img"]
mu_txt = data["mu_txt"]
mu_shared = data["mu_shared"]

diff = mu_shared - (mu_img + mu_txt) / 2
print("Max diff between mu_shared and (mu_img + mu_txt) / 2:", np.max(np.abs(diff)))
print("mu_shared norm:", np.linalg.norm(mu_shared))
print("mu_img norm:", np.linalg.norm(mu_img))
print("mu_txt norm:", np.linalg.norm(mu_txt))
print("cone_restore_gamma:", data["cone_restore_gamma"])
print("csls_beta:", data["csls_beta"])
print("csls_k:", data["csls_k"])
print("gallery_hub_scores shape and stats:", data["gallery_hub_scores"].shape, "min:", data["gallery_hub_scores"].min(), "mean:", data["gallery_hub_scores"].mean(), "max:", data["gallery_hub_scores"].max())
