from __future__ import annotations

from collections.abc import Sequence

from vit_pytorch.vit_3d import ViT


def _vision_transformer(
    image_size: Sequence[int],
    patch_size: Sequence[int],
    num_layers: int,
    num_heads: int,
    hidden_size: int,
    mlp_dim: int,
) -> ViT:


    model = ViT(
        image_size = image_size[1],          # image size
        frames = image_size[0],               # number of frames
        image_patch_size = patch_size[1],     # image patch size
        frame_patch_size = patch_size[0],      # frame patch size
        num_classes = 1000,
        dim = hidden_size,
        depth = num_layers,
        heads = num_heads,
        mlp_dim = mlp_dim,
        dropout = 0.1,
        emb_dropout = 0.1,
        channels = 1
    )

    return model

def vit_b_32() -> ViT:
    """
    Constructs a vit_b_32 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.
    """

    return _vision_transformer(
        patch_size=(32, 32, 32),
        num_layers=12,
        num_heads=12,
        hidden_size=768,
        mlp_dim=3072,
        image_size=(192, 192, 192)
    )


def vit_b_16() -> ViT:
    """
    Constructs a vit_b_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.
    """

    return _vision_transformer(
        patch_size=(16, 16, 16),
        num_layers=12,
        num_heads=12,
        hidden_size=768,
        mlp_dim=3072,
        image_size=(192, 192, 192)
    )



def vit_l_16() -> ViT:
    """
    Constructs a vit_l_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.
    """

    return _vision_transformer(
        patch_size=(16, 16, 16),
        num_layers=24,
        num_heads=16,
        hidden_size=1024,
        mlp_dim=4096,
        image_size=(192, 192, 192)
    )


def vit_l_32() -> ViT:
    """
    Constructs a vit_l_32 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.
    """
    return _vision_transformer(
        patch_size=(32, 32, 32),
        num_layers=24,
        num_heads=16,
        hidden_size=1024,
        mlp_dim=4096,
        image_size=(192, 192, 192)
    )


def vit_h_14() -> ViT:
    """
    Constructs a vit_h_14 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.
    """

    return _vision_transformer(
        patch_size=(14, 14, 14),
        num_layers=32,
        num_heads=16,
        hidden_size=1280,
        mlp_dim=5120,
        image_size=(192, 192, 192)
    )

