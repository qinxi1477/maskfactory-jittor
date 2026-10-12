import jittor as jt

from masactrl.jittor_compat import enable_cuda
from masactrl.masactrl import Discriminator, Generator, MutualSelfAttentionControl
from masactrl.masactrl_utils import AttentionBase


def main():
    enable_cuda()
    image = jt.randn((1, 3, 512, 512))

    generator = Generator()
    generated = generator(image)
    assert tuple(generated.shape) == (1, 3, 512, 512), generated.shape

    discriminator = Discriminator()
    score = discriminator(generated)
    assert score.shape[0] == 1, score.shape

    num_heads = 2
    q = jt.randn((4, 16, 8))
    k = jt.randn((4, 16, 8))
    v = jt.randn((4, 16, 8))
    attn = jt.nn.softmax(jt.einsum("b i d, b j d -> b i j", q, k), dim=-1)

    base = AttentionBase()
    base.num_att_layers = 1
    out = base(q, k, v, None, attn, False, "mid", num_heads)
    assert tuple(out.shape) == (2, 16, 16), out.shape

    editor = MutualSelfAttentionControl(start_step=0, start_layer=0, total_steps=1)
    editor.num_att_layers = 1
    out = editor(q, k, v, None, attn, False, "mid", num_heads, scale=1.0)
    assert tuple(out.shape) == (2, 16, 16), out.shape

    jt.sync_all()
    print("smoke_jittor passed")


if __name__ == "__main__":
    main()
