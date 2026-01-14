import torch

from ivf.models.encoder_embryonet_lite import EmbryoNetLite


def test_embryonet_lite_forward_and_params():
    model = EmbryoNetLite(feature_dim=512, width_mult=1.0)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, model.out_dim)
    assert model.out_dim == 512
    params = sum(p.numel() for p in model.parameters())
    print(f"EmbryoNetLite params: {params}")
    assert params < 6_000_000


def test_embryonet_lite_no_pretrained_load():
    orig_load = torch.load
    called = {"flag": False}

    def _fail_load(*args, **kwargs):
        called["flag"] = True
        raise RuntimeError("torch.load should not be called for EmbryoNetLite.")

    torch.load = _fail_load
    try:
        model = EmbryoNetLite(feature_dim=512, width_mult=1.0)
        x = torch.randn(1, 3, 224, 224)
        _ = model(x)
    finally:
        torch.load = orig_load

    assert called["flag"] is False
