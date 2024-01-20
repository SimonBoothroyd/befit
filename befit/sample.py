"""Generate samples to train against."""
import collections
import tempfile
import typing

import geometric.engine
import geometric.internal
import geometric.molecule
import geometric.prepare
import geometric.run_json
import numpy
import openff.toolkit
import openff.units
import openmm
import openmm.app
import openmm.unit
import torsiondrive.td_api


class Sample(typing.NamedTuple):
    """The energy and force of a molecule in a specific conformation."""

    coords: openmm.unit.Quantity

    energy: openmm.unit.Quantity
    forces: openmm.unit.Quantity | None = None


class TDSample(typing.NamedTuple):
    """The energy and force of a molecule at a given torsion scan angle."""

    angle: float
    coords: openmm.unit.Quantity

    energy: openmm.unit.Quantity
    forces: openmm.unit.Quantity | None = None


class _OpenMMEngine(geometric.engine.Engine):
    """A wrapper for GeomeTric that allows using an existing OpenMM system to evaluate
    the energy and gradient of a molecule."""

    def __init__(self, molecule, system: openmm.System):
        self.context = openmm.Context(
            system,
            openmm.VerletIntegrator(1.0),
            openmm.Platform.getPlatformByName("Reference"),
        )
        super(_OpenMMEngine, self).__init__(molecule)

    def calc_new(self, coords, dirname):
        self.context.setPositions(coords.reshape(-1, 3) * openmm.unit.bohr)

        state = self.context.getState(getEnergy=True, getForces=True)

        energy = state.getPotentialEnergy() / openmm.unit.AVOGADRO_CONSTANT_NA
        gradient = -state.getForces(asNumpy=True) / openmm.unit.AVOGADRO_CONSTANT_NA

        return {
            "energy": energy.value_in_unit(openmm.unit.hartree),
            "gradient": gradient.value_in_unit(
                openmm.unit.hartree / openmm.unit.bohr
            ).flatten(),
        }


def generate_sp_sample(system: openmm.System, coords: openmm.unit.Quantity) -> Sample:
    """Perform a single-point energy evaluation using a given OpenMM system.

    Args:
        system: The OpenMM system to use to evaluate the energy / force.
        coords: The coordinates to evaluate at.

    Returns:
        The energy and forces of the molecule at the specified coordinates.
    """

    context = openmm.Context(
        system,
        openmm.VerletIntegrator(1.0),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(coords)

    state = context.getState(getEnergy=True, getForces=True)

    energy = state.getPotentialEnergy()
    forces = state.getForces(asNumpy=True)

    return Sample(coords, energy, forces)


def generate_md_samples(
    mol: openff.toolkit.Molecule,
    system: openmm.System,
    coords: openmm.unit.Quantity,
    temperature: openmm.unit.Quantity,
    n_steps_per_sample: int,
    n_samples: int,
    n_warmup_steps: int = 0,
    timestep: openmm.unit.Quantity = 1.0 * openmm.unit.femtosecond,
    friction: openmm.unit.Quantity = 1.0 / openmm.unit.picosecond,
) -> list[Sample]:
    """Perform a molecular dynamics simulation of a molecule using OpenMM, and
    returns the sampled conformations / energies.

    Args:
        mol: The molecule to simulate.
        system: The OpenMM system to use to evaluate the energy / force.
        coords: The initial coordinates to simulate.
        temperature: The temperature to simulate at.
        n_steps_per_sample: The number of steps to simulate for between each sample.
        n_samples: The number of samples to generate.
        n_warmup_steps: The number of steps to simulate for before collecting samples.
        timestep: The timestep to use during the simulation.
        friction: The friction coefficient to use during the simulation.

    Returns:
        The sampled conformations and energies.
    """
    integrator = openmm.LangevinMiddleIntegrator(temperature, friction, timestep)

    simulation = openmm.app.Simulation(
        mol.to_topology().to_openmm(), system, integrator
    )
    simulation.context.setPositions(coords)

    simulation.step(n_warmup_steps)

    samples = []

    for _ in range(n_samples):
        simulation.step(n_steps_per_sample)

        state = simulation.context.getState(
            getPositions=True, getEnergy=True, getForces=True
        )

        sample = Sample(
            state.getPositions(numpy=True),
            state.getPotentialEnergy(),
            state.getForces(numpy=True),
        )
        samples.append(sample)

    return samples


def _optimize_geometric(
    mol: openff.toolkit.Molecule,
    system: openmm.System,
    coords: openmm.unit.Quantity,
    constraints: dict[str, typing.Any],
) -> tuple[openmm.unit.Quantity, openmm.unit.Quantity]:
    """Perform a geometry optimization of a molecule using GeomeTRIC, keeping a
    specified dihedral angle fixed."""

    with tempfile.NamedTemporaryFile(suffix=".pdb") as file:
        mol.to_file(file.name, "PDB")

        mol_tric = geometric.molecule.Molecule(file.name, radii={}, fragment=False)
        mol_tric.xyzs = [coords.value_in_unit(openmm.unit.angstrom)]

    constraint_strings, constraint_values = geometric.prepare.parse_constraints(
        mol_tric, geometric.run_json.make_constraints_string(constraints)
    )

    params = geometric.optimize.OptParams()
    engine = _OpenMMEngine(mol_tric, system)

    for _, constraint_value in enumerate(constraint_values):
        internal_coords = geometric.internal.DelocalizedInternalCoordinates(
            mol_tric,
            build=True,
            connect=True,
            addcart=False,
            constraints=constraint_strings,
            cvals=constraint_value,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = geometric.optimize.Optimize(
                coords.value_in_unit(openmm.unit.bohr).flatten(),
                mol_tric,
                internal_coords,
                engine,
                tmp_dir,
                params,
                False,
            )

    coords_final = result.xyzs[-1].flatten() * openmm.unit.angstrom
    energy_final = result.qm_energies[-1] * openmm.unit.hartree

    return coords_final, energy_final


def _bond_to_dihedral_idxs(
    mol: openff.toolkit.Molecule, bond: tuple[int, int]
) -> tuple[int, int, int, int]:
    idx_2, idx_3 = bond

    atoms_1 = [
        atom
        for atom in mol.atoms[idx_2].bonded_atoms
        if atom.molecule_atom_index != idx_3
    ]
    idx_1 = max(atoms_1, key=lambda atom: atom.atomic_number).molecule_atom_index

    atoms_4 = [
        atom
        for atom in mol.atoms[idx_3].bonded_atoms
        if atom.molecule_atom_index != idx_2
    ]
    idx_4 = max(atoms_4, key=lambda atom: atom.atomic_number).molecule_atom_index

    return idx_1, idx_2, idx_3, idx_4


def generate_td_samples(
    mol: openff.toolkit.Molecule,
    system: openmm.System,
    coords: openmm.unit.Quantity,
    bond: tuple[int, int],
    grid_spacing: int = 15,
    energy_decrease_thresh: float | None = None,
    energy_upper_limit: float | None = None,
) -> list[TDSample]:
    """Perform a torsion scan of a molecule using GeomeTRIC

    Args:
        mol: The molecule to scan.
        system: The OpenMM system to use to evaluate the energy / force.
        coords: The initial coordinates to scan from.
        bond: The bond to scan around.
        grid_spacing: The spacing between each grid point.
        energy_decrease_thresh: The minimum energy decrease required to continue
            scanning.
        energy_upper_limit: The maximum energy to scan up to.
    """

    dihedral = _bond_to_dihedral_idxs(mol, bond)

    state = torsiondrive.td_api.create_initial_state(
        dihedrals=[dihedral],
        grid_spacing=[grid_spacing],
        elements=[atom.symbol for atom in mol.atoms],
        init_coords=[coords.value_in_unit(openmm.unit.bohr).flatten().tolist()],
        dihedral_ranges=None,
        energy_upper_limit=energy_upper_limit,
        energy_decrease_thresh=energy_decrease_thresh,
    )

    optimization_results = collections.defaultdict(list)

    while True:
        next_jobs = torsiondrive.td_api.next_jobs_from_state(state, verbose=False)

        if len(next_jobs) == 0:
            break

        grid_point_results = {}

        for angle, jobs in next_jobs.items():
            constraints = {
                "set": [{"indices": dihedral, "type": "dihedral", "value": angle}]
            }
            job_results = []

            for coords_start_flat in jobs:
                coords_start = (
                    numpy.array(coords_start_flat).reshape(-1, 3) * openmm.unit.bohr
                )

                coords_final, energy_final = _optimize_geometric(
                    mol, system, coords_start, constraints
                )
                coords_final_flat = (
                    coords_final.value_in_unit(openmm.unit.bohr).flatten().tolist()
                )

                energy_final = energy_final.value_in_unit(openmm.unit.hartree)

                job_results.append((coords_start_flat, coords_final_flat, energy_final))

            grid_point_results[angle] = job_results
            optimization_results[angle].extend(job_results)

        torsiondrive.td_api.update_state(state, {**grid_point_results})

    samples = []

    for angle, results in optimization_results.items():
        final_energies = numpy.array([energy for _, _, energy in results])

        lowest_energy_idx = final_energies.argmin()
        lowest_energy_result = results[lowest_energy_idx]

        coords = numpy.array(lowest_energy_result[1]).reshape(-1, 3) * openmm.unit.bohr
        coords = coords.in_units_of(openmm.unit.angstrom)

        _, energy, forces = generate_sp_sample(system, coords)

        samples.append(TDSample(float(angle), coords, energy, forces))

    return samples


def generate_opt_sample(
    mol: openff.toolkit.Molecule,
    system: openmm.System,
    coords: openmm.unit.Quantity,
    freeze_dihedrals: bool = False,
) -> Sample:
    """Perform a geometry optimization of a molecule using GeomeTRIC, returning
    the optimized energy and forces.

    Args:
        mol: The molecule to optimize.
        system: The OpenMM system to use to evaluate the energy / force.
        coords: The initial coordinates to optimize.
        freeze_dihedrals: Whether to freeze the dihedral angles of the molecule during
            the optimization.

    Returns:
        The optimized energy and forces of the molecule.
    """
    constraints = {}

    if freeze_dihedrals:
        dihedrals = [[a.molecule_atom_index for a in proper] for proper in mol.propers]

        constraints = {
            "freeze": [{"indices": idxs, "type": "dihedral"} for idxs in dihedrals]
        }

    coords, _ = _optimize_geometric(mol, system, coords, constraints)
    return generate_sp_sample(system, coords)
