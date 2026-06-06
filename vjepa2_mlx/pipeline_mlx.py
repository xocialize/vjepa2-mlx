"""Embedding/prediction pipeline + CLI. P5.
Video/image -> preprocess (256, 64fpc, normalize) -> encoder -> embeddings;
predictor for JEPA latent prediction; attentive pooler for classification."""
def embed_video(path, checkpoint="vitl-encoder"):
    raise NotImplementedError("P5")
def cli_main():
    raise NotImplementedError("P5")
if __name__ == "__main__":
    cli_main()
