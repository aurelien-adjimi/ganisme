from pathlib import Path
import json

import streamlit as st
import torch
from PIL import Image

from src.models.generator import Generator


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models/final"

st.set_page_config(
    page_title="GAN Art Generator",
    page_icon="🎨",
)

st.title("🎨 GAN Art Generator")
st.write("Générez des images à partir d’un vecteur de bruit aléatoire.")
st.caption(
    "Prototype pédagogique : les générations restent abstraites "
    "et bruitées. Résolution native : 32 × 32 pixels."
)


@st.cache_resource
def load_generator():
    selection = json.loads(
        (MODEL_DIR / "selection.json").read_text(encoding="utf-8")
    )
    latent_dim = selection["config"]["latent_dim"]

    model = Generator(latent_dim=latent_dim)

    weights = torch.load(
        MODEL_DIR / "generator.pt",
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(weights)
    model.eval()

    return model, latent_dim


if not (
    (MODEL_DIR / "generator.pt").is_file()
    and (MODEL_DIR / "selection.json").is_file()
):
    st.error(
        "Modèle absent : exportez generator.pt et selection.json "
        "dans models/final depuis le notebook."
    )
    st.stop()

model, latent_dim = load_generator()

number = st.slider(
    "Nombre d’images",
    min_value=1,
    max_value=9,
    value=1,
)

button_label = (
    "Générer à nouveau"
    if "artworks" in st.session_state
    else "Générer"
)

if st.button(button_label, type="primary"):
    with st.spinner("Génération…"):
        noise = torch.randn(number, latent_dim)

        with torch.inference_mode():
            images = model(noise)

        # Annuler la normalisation pour afficher les pixels.
        pixels = (
            ((images + 1) / 2)
            .clamp(0, 1)
            .permute(0, 2, 3, 1)
            .mul(255)
            .round()
            .to(torch.uint8)
            .numpy()
        )

        st.session_state["artworks"] = [
            Image.fromarray(array) for array in pixels
        ]

if "artworks" in st.session_state:
    artworks = st.session_state["artworks"]
    columns = st.columns(min(3, len(artworks)))

    for index, artwork in enumerate(artworks):
        # Agrandissement d’affichage sans ajouter de détails.
        preview = artwork.resize(
            (256, 256),
            resample=Image.Resampling.NEAREST,
        )

        with columns[index % len(columns)]:
            st.image(
                preview,
                caption=f"Génération {index + 1}",
            )