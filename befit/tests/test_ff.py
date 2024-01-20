import copy

import openff.interchange
import openff.toolkit
import pytest
import smee
import smee.converters
import torch

import befit.ff


@pytest.mark.parametrize(
    "smiles",
    ["CC", "ClC=O", "OCCO", "CC(=O)Nc1ccc(cc1)O"],
)
def test_prepare(smiles):
    mol = openff.toolkit.Molecule.from_smiles(smiles)
    mol.generate_conformers(n_conformers=1)

    coords = torch.tensor(mol.conformers[0].m_as("angstrom"))
    coords += torch.randn(len(coords) * 3).reshape(-1, 3) * 0.5

    base = openff.toolkit.ForceField("openff_unconstrained-2.1.0.offxml")

    ff, [top] = smee.converters.convert_interchange(
        openff.interchange.Interchange.from_smirnoff(base, mol.to_topology())
    )
    energy_0 = smee.compute_energy(top, ff, coords)

    ff_1, top_1 = befit.ff.prepare(mol, base)
    energy_1 = smee.compute_energy(top_1, ff_1, coords)

    assert torch.isclose(energy_0, energy_1)

    base_no_valence = copy.deepcopy(base)
    base_no_valence.deregister_parameter_handler("Bonds")
    base_no_valence.deregister_parameter_handler("Angles")
    base_no_valence.deregister_parameter_handler("ProperTorsions")
    base_no_valence.deregister_parameter_handler("ImproperTorsions")

    converted = befit.ff.convert_to_smirnoff(ff_1, base_no_valence)

    ff_2, [top_2] = smee.converters.convert_interchange(
        openff.interchange.Interchange.from_smirnoff(converted, mol.to_topology())
    )
    energy_2 = smee.compute_energy(top_2, ff_2, coords)

    assert torch.isclose(energy_0, energy_2)
