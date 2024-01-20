"""Utilities for working with AIMNet2."""
import tempfile
import typing
import urllib.request

import openff.toolkit
import openff.units
import openmm
import openmmtorch
import torch

_GIT_COMMIT = "854adf3"
_GIT_URL = f"https://github.com/isayevlab/AIMNet2/raw/{_GIT_COMMIT}/models/aimnet2"

DEFAULT_MODEL_METHOD = "wb97m-d3"
DEFAULT_MODEL_VERSION = 0


def _download_model(
    method: str, version: int | typing.Literal["ens"], device: str | None
):
    """Download an AIMNet2 model directly from GitHub."""
    url = f"{_GIT_URL}_{method}_{version}.jpt"

    with tempfile.NamedTemporaryFile(suffix=".jpt") as tmp_file:
        urllib.request.urlretrieve(url, filename=tmp_file.name)
        return torch.jit.load(tmp_file.name, map_location=device)


class _AIMNET2Wrapper(torch.nn.Module):
    """A wrapper around an AIMNet2 model that can be used as an OpenMM force."""

    def __init__(
        self,
        atomic_numbers: list[int],
        charge: int,
        method: str,
        version: int,
        device: str | None = None,
    ):
        super().__init__()

        self.atomic_numbers = torch.tensor(atomic_numbers).unsqueeze(0)
        self.charge = torch.tensor([charge], device=device)

        self.model = _download_model(method, version, device)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        _EV_TO_KJ_PER_MOL = 96.48533212331
        _NM_TO_ANGSTROM = 10.0

        positions = positions.unsqueeze(0).float() * _NM_TO_ANGSTROM

        result = self.model(
            {"numbers": self.atomic_numbers, "coord": positions, "charge": self.charge}
        )
        return result["energy"].squeeze(0) * _EV_TO_KJ_PER_MOL


def create_force(
    mol: openff.toolkit.Molecule,
    method: typing.Literal["wb97m-d3", "b973c"] = DEFAULT_MODEL_METHOD,
    version: int = DEFAULT_MODEL_VERSION,
) -> openmmtorch.TorchForce:
    """Create an OpenMM force which evaluates the energy of molecule using
    AIMNet2.

    Args:
        mol: The molecule to create the force for.
        method: The level of QM theory the model was trained at.
        version: The version of the model to use.

    Returns:
        The OpenMM force containing the AIMNet2 representation of the molecule.
    """

    atomic_numbers = [atom.atomic_number for atom in mol.atoms]
    charge = mol.total_charge.m_as(openff.units.unit.e)

    wrapper = _AIMNET2Wrapper(
        atomic_numbers=atomic_numbers, charge=charge, method=method, version=version
    )

    force = openmmtorch.TorchForce(torch.jit.script(wrapper))
    return force


def create_system(
    mol: openff.toolkit.Molecule,
    method: typing.Literal["wb97m-d3", "b973c"] = DEFAULT_MODEL_METHOD,
    version: int = DEFAULT_MODEL_VERSION,
) -> openmm.System:
    """Create an OpenMM system that contains a AIMNet2 force.

    See Also:
        ``befit.utils.aimnet2.create_force``

    Args:
        mol: The molecule to create the force for.
        method: The level of QM theory the model was trained at.
        version: The version of the model to use.

    Returns:
        The OpenMM system containing the AIMNet2 force.
    """

    system = openmm.System()
    system.addForce(openmm.CMMotionRemover())

    for atom in mol.atoms:
        system.addParticle(atom.mass.to_openmm())

    system.addForce(create_force(mol, method, version))

    return system
