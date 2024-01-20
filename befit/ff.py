"""Prepare force fields for bespoke parameter fitting."""
import collections
import copy

import openff.interchange
import openff.interchange.models
import openff.toolkit
import smee
import smee.converters
import smee.utils
import torch

_SUPPORTED_HANDLERS = {"Bonds", "Angles", "ProperTorsions", "ImproperTorsions"}


def _create_smarts(mol: openff.toolkit.Molecule, idxs: torch.Tensor) -> str:
    """Create a mapped SMARTS representation of a molecule."""
    from rdkit import Chem

    mol_rdkit = mol.to_rdkit()

    for i, idx in enumerate(idxs):
        atom = mol_rdkit.GetAtomWithIdx(int(idx))
        atom.SetAtomMapNum(i + 1)

    smarts = Chem.MolToSmarts(mol_rdkit)
    return smarts


def _get_atom_symmetries(mol: openff.toolkit.Molecule) -> list[int]:
    """Get the topological symmetry indices of each atom in a molecule."""
    from rdkit import Chem

    rd_mol = mol.to_rdkit()
    return list(Chem.CanonicalRankAtoms(rd_mol, breakTies=False))


def _prepare_potential(
    mol: openff.toolkit.Molecule,
    symmetries: list[int],
    potential: smee.TensorPotential,
    parameter_map: smee.ParameterMap,
) -> None:
    """Prepare a potential to use bespoke parameters for each 'slot'."""

    is_indexed = any(key.mult is not None for key in potential.parameter_keys)

    ids_to_parameter_idxs = collections.defaultdict(set)
    ids_to_particle_idxs = collections.defaultdict(set)

    ids_to_smarts = {}

    for particle_idxs, assignment_row in zip(
        parameter_map.particle_idxs,
        parameter_map.assignment_matrix.to_dense(),
        strict=True,
    ):
        particle_idxs = tuple(int(idx) for idx in particle_idxs)
        particle_ids = tuple(symmetries[idx] for idx in particle_idxs)

        if potential.type != "ImproperTorsions" and particle_ids[-1] < particle_ids[0]:
            particle_ids = particle_ids[::-1]

        parameter_idxs = [
            parameter_idx
            for parameter_idx, value in enumerate(assignment_row)
            if int(value) != 0
        ]
        assert len(parameter_idxs) == 1

        ids_to_parameter_idxs[particle_ids].add(parameter_idxs[0])
        ids_to_particle_idxs[particle_ids].add(particle_idxs)

        if potential.type == "ImproperTorsions":
            particle_idxs = (
                particle_idxs[1],
                particle_idxs[0],
                particle_idxs[2],
                particle_idxs[3],
            )

        ids_to_smarts[particle_ids] = _create_smarts(mol, particle_idxs)

    ids_to_parameter_idxs = {
        particle_ids: sorted(parameter_idxs)
        for particle_ids, parameter_idxs in ids_to_parameter_idxs.items()
    }

    parameter_ids = [
        (particle_ids, parameter_idx)
        for particle_ids, parameter_idxs in ids_to_parameter_idxs.items()
        for parameter_idx in parameter_idxs
    ]
    potential.parameters = potential.parameters[
        [parameter_idx for _, parameter_idx in parameter_ids]
    ]
    potential.parameter_keys = [
        openff.interchange.models.PotentialKey(
            id=ids_to_smarts[particle_ids],
            mult=ids_to_parameter_idxs[particle_ids].index(parameter_idx)
            if is_indexed
            else None,
            associated_handler=potential.type,
        )
        for particle_ids, parameter_idx in parameter_ids
    ]

    assignment_matrix = smee.utils.zeros_like(
        (len(parameter_map.particle_idxs), len(potential.parameters)),
        parameter_map.assignment_matrix,
    )
    particle_idxs_updated = []

    for particle_ids, particle_idxs in ids_to_particle_idxs.items():
        for particle_idx in particle_idxs:
            for parameter_idx in ids_to_parameter_idxs[particle_ids]:
                j = parameter_ids.index((particle_ids, parameter_idx))

                assignment_matrix[len(particle_idxs_updated), j] = 1
                particle_idxs_updated.append(particle_idx)

    parameter_map.particle_idxs = smee.utils.tensor_like(
        particle_idxs_updated, parameter_map.particle_idxs
    )
    parameter_map.assignment_matrix = assignment_matrix.to_sparse()


def prepare(
    mol: openff.toolkit.Molecule, base: openff.toolkit.ForceField
) -> tuple[smee.TensorForceField, smee.TensorTopology]:
    """Prepare a tensor force field that contains unique parameters for each
    topologically symmetric term of a molecule.

    Notes:
        The function only copies the parameter values from the ``base`` force field,
        and does not try and refit them.

    Args:
        mol: The molecule to prepare bespoke parameters for.
        base: The base force field to copy the parameters from.

    Returns:
        The prepared tensor force field and topology ready for fitting.
    """

    ff, [top] = smee.converters.convert_interchange(
        openff.interchange.Interchange.from_smirnoff(base, mol.to_topology())
    )

    symmetries = _get_atom_symmetries(mol)

    if top.n_v_sites != 0:
        raise NotImplementedError("virtual sites are not supported yet.")

    for potential in ff.potentials:
        parameter_map = top.parameters[potential.type]

        if isinstance(parameter_map, smee.NonbondedParameterMap):
            continue

        _prepare_potential(mol, symmetries, potential, parameter_map)

    return ff, top


def convert_to_smirnoff(
    ff: smee.TensorForceField, base: openff.toolkit.ForceField | None = None
) -> openff.toolkit.ForceField:
    """Convert a tensor force field that *contains bespoke valence parameters* to
    SMIRNOFF format.

    Notes:
        * Currently only the valence terms are converted into SMIRNOFF format.
        * Currently only bond, angle, torsion and improper potentials are supported.

    See Also:
        `befit.ff.prepare`

    Args:
        ff: The force field containing the bespoke valence terms.
        base: The (optional) original SMIRNOFF force field to add the bespoke
            parameters to. If no specified, a force field containing only the bespoke
            parameters will be returned.

    Returns:
        A SMIRNOFF force field containing the valence terms of the input force field.
    """
    ff_smirnoff = openff.toolkit.ForceField() if base is None else copy.deepcopy(base)

    for potential in ff.potentials:
        if potential.type not in _SUPPORTED_HANDLERS:
            continue

        assert potential.attribute_cols is None

        parameters_by_smarts = collections.defaultdict(dict)

        for parameter, parameter_key in zip(
            potential.parameters, potential.parameter_keys, strict=True
        ):
            assert parameter_key.mult not in parameters_by_smarts[parameter_key.id]
            parameters_by_smarts[parameter_key.id][parameter_key.mult] = parameter

        handler = ff_smirnoff.get_parameter_handler(potential.type)

        for smarts, parameters_by_mult in parameters_by_smarts.items():
            mults = {*parameters_by_mult}

            if None in mults and len(mults) > 1:
                raise NotImplementedError("unexpected parameters found")

            if None not in mults and mults != {*range(len(mults))}:
                raise NotImplementedError("unexpected parameters found")

            counter = len(handler.parameters) + 1

            parameter_id = f"{potential.type[0].lower()}-bespoke-{counter}"

            parameter_dict = {"smirks": smarts, "id": parameter_id}
            parameter_dict.update(
                {
                    (col if mult is None else f"{col}{mult + 1}"): float(
                        parameter[col_idx]
                    )
                    * potential.parameter_units[col_idx]
                    for mult, parameter in parameters_by_mult.items()
                    for col_idx, col in enumerate(potential.parameter_cols)
                }
            )
            handler.add_parameter(parameter_dict)

    return ff_smirnoff
