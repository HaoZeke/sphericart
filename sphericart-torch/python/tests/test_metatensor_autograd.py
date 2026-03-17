"""Tests that autograd flows through the sphericart.torch.metatensor wrapper.

The existing test_metatensor.py only checks value correctness. This module
verifies that torch autograd (backward, gradcheck) works end-to-end when
spherical harmonics are computed through the metatensor TensorMap wrapper.
"""

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

import sphericart.torch
import sphericart.torch.metatensor


L_MAX = 4
N_SAMPLES = 20


def _make_xyz_tensormap(xyz_raw: torch.Tensor) -> TensorMap:
    """Wrap a (N, 3) tensor into the TensorMap format expected by the wrapper."""
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=xyz_raw.unsqueeze(-1),  # (N, 3, 1)
                samples=Labels(
                    names=["sample"],
                    values=torch.arange(xyz_raw.shape[0]).reshape(-1, 1),
                ),
                components=[
                    Labels(names=["xyz"], values=torch.arange(3).reshape(-1, 1))
                ],
                properties=Labels.single(),
            )
        ],
    )


def test_compute_preserves_grad_fn():
    """Values in output TensorBlocks should have grad_fn when input requires_grad."""
    xyz_raw = torch.randn(N_SAMPLES, 3, dtype=torch.float64, requires_grad=True)
    xyz_map = _make_xyz_tensormap(xyz_raw)

    calculator = sphericart.torch.metatensor.SphericalHarmonics(L_MAX)
    result = calculator.compute(xyz_map)

    for block in result.blocks():
        assert block.values.requires_grad, (
            "TensorBlock values should require grad"
        )
        assert block.values.grad_fn is not None, (
            "TensorBlock values should have grad_fn"
        )


def test_backward_through_tensormap():
    """Backward pass through metatensor-wrapped output should produce gradients."""
    xyz_raw = torch.randn(N_SAMPLES, 3, dtype=torch.float64, requires_grad=True)
    xyz_map = _make_xyz_tensormap(xyz_raw)

    calculator = sphericart.torch.metatensor.SphericalHarmonics(L_MAX)
    result = calculator.compute(xyz_map)

    # Sum all block values and backpropagate
    loss = sum(block.values.sum() for block in result.blocks())
    loss.backward()

    assert xyz_raw.grad is not None, "Input should have gradients after backward"
    assert not torch.all(xyz_raw.grad == 0), "Gradients should be non-zero"


def test_backward_matches_raw():
    """Gradients through metatensor wrapper should match raw sphericart-torch."""
    xyz_raw = torch.randn(N_SAMPLES, 3, dtype=torch.float64, requires_grad=True)
    xyz_raw_copy = xyz_raw.detach().clone().requires_grad_(True)

    # Metatensor path
    xyz_map = _make_xyz_tensormap(xyz_raw)
    calculator_mts = sphericart.torch.metatensor.SphericalHarmonics(L_MAX)
    result = calculator_mts.compute(xyz_map)
    loss_mts = sum(block.values.sum() for block in result.blocks())
    loss_mts.backward()

    # Raw path
    calculator_raw = sphericart.torch.SphericalHarmonics(L_MAX)
    sh_raw = calculator_raw.compute(xyz_raw_copy)
    loss_raw = sh_raw.sum()
    loss_raw.backward()

    assert torch.allclose(xyz_raw.grad, xyz_raw_copy.grad, atol=1e-12), (
        f"Gradient mismatch: max diff = {(xyz_raw.grad - xyz_raw_copy.grad).abs().max()}"
    )


def test_gradcheck_compute():
    """Numerical gradient check through the metatensor wrapper."""
    torch.manual_seed(42)
    xyz_raw = torch.randn(N_SAMPLES, 3, dtype=torch.float64, requires_grad=True)

    calculator = sphericart.torch.metatensor.SolidHarmonics(
        L_MAX, backward_second_derivatives=True
    )

    def fn(xyz_in):
        xyz_map = _make_xyz_tensormap(xyz_in)
        result = calculator.compute(xyz_map)
        # Return concatenated values from all blocks
        return torch.cat([block.values.reshape(N_SAMPLES, -1) for block in result.blocks()], dim=-1)

    assert torch.autograd.gradcheck(fn, xyz_raw, fast_mode=True)


def test_gradcheck_compute_second_derivatives():
    """Double backward (gradgradcheck) through the metatensor wrapper."""
    torch.manual_seed(42)
    xyz_raw = torch.randn(N_SAMPLES, 3, dtype=torch.float64, requires_grad=True)

    calculator = sphericart.torch.metatensor.SolidHarmonics(
        L_MAX, backward_second_derivatives=True
    )

    def fn(xyz_in):
        xyz_map = _make_xyz_tensormap(xyz_in)
        result = calculator.compute(xyz_map)
        return torch.cat([block.values.reshape(N_SAMPLES, -1) for block in result.blocks()], dim=-1)

    assert torch.autograd.gradgradcheck(fn, xyz_raw, fast_mode=True)


def test_solid_harmonics_autograd():
    """SolidHarmonics metatensor wrapper also supports autograd."""
    xyz_raw = torch.randn(N_SAMPLES, 3, dtype=torch.float64, requires_grad=True)
    xyz_map = _make_xyz_tensormap(xyz_raw)

    calculator = sphericart.torch.metatensor.SolidHarmonics(L_MAX)
    result = calculator.compute(xyz_map)

    loss = sum(block.values.sum() for block in result.blocks())
    loss.backward()

    assert xyz_raw.grad is not None
    assert not torch.all(xyz_raw.grad == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_autograd():
    """Autograd works on CUDA through the metatensor wrapper."""
    xyz_raw = torch.randn(N_SAMPLES, 3, dtype=torch.float64, device="cuda", requires_grad=True)
    xyz_map = _make_xyz_tensormap(xyz_raw)

    calculator = sphericart.torch.metatensor.SphericalHarmonics(L_MAX)
    result = calculator.compute(xyz_map)

    loss = sum(block.values.sum() for block in result.blocks())
    loss.backward()

    assert xyz_raw.grad is not None
    assert not torch.all(xyz_raw.grad == 0)
