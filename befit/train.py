"""Train the force field."""
import contextlib
import math
import pathlib
import tempfile
import typing

import descent.targets.energy
import openmm
import pydantic
import smee
import tensorboardX
import torch

if typing.TYPE_CHECKING:
    import befit.sample


class ParameterConfig(pydantic.BaseModel):
    """Configuration for how a potential's parameters should be trained."""

    cols: list[str] = pydantic.Field(
        description="The parameters to train, e.g. 'k', 'length', 'epsilon'."
    )

    scales: dict[str, float] = pydantic.Field(
        {},
        description="The scales to apply to each parameter, e.g. 'k': 1.0, "
        "'length': 1.0, 'epsilon': 1.0.",
    )
    constraints: dict[str, tuple[float | None, float | None]] = pydantic.Field(
        {},
        description="The min and max values to clamp each parameter within, e.g. "
        "'k': (0.0, None), 'angle': (0.0, pi), 'epsilon': (0.0, None), where "
        "none indicates no constraint.",
    )


_DEFAULT_PARAMS = {
    "Bonds": ParameterConfig(
        cols=["k", "length"],
        scales={"k": 1.0 / 100.0, "length": 1.0},
        constraints={"k": (0.0, None), "length": (0.0, None)},
    ),
    "Angles": ParameterConfig(
        cols=["k", "angle"],
        scales={"k": 1.0 / 100.0, "angle": 1.0},
        constraints={"k": (0.0, None), "angle": (0.0, math.pi)},
    ),
    "ProperTorsions": ParameterConfig(cols=["k"], scales={"k": 1.0}),
    # "ImproperTorsions": ParameterConfig(cols=["k"], scales={"k": 1.0}),
}


class _Trainable:
    """A wrapper around a SMEE force field that handles zeroing out gradients of
    fixed parameters and applying parameter constraints."""

    def __init__(
        self,
        force_field: smee.TensorForceField,
        parameters: dict[str, ParameterConfig],
    ):
        self.potential_types = [*parameters]
        self._force_field = force_field

        potentials = [
            force_field.potentials_by_type[potential_type]
            for potential_type in self.potential_types
        ]

        self._frozen_cols = [
            [
                i
                for i, col in enumerate(potential.parameter_cols)
                if col not in parameters[potential_type].cols
            ]
            for potential_type, potential in zip(
                self.potential_types, potentials, strict=True
            )
        ]

        self._scales = [
            torch.tensor(
                [
                    parameters[potential_type].scales.get(col, 1.0)
                    for col in potential.parameter_cols
                ]
            ).reshape(1, -1)
            for potential_type, potential in zip(
                self.potential_types, potentials, strict=True
            )
        ]
        self._constraints = [
            {
                i: parameters[potential_type].constraints[col]
                for i, col in enumerate(potential.parameter_cols)
                if col in parameters[potential_type].constraints
            }
            for potential_type, potential in zip(
                self.potential_types, potentials, strict=True
            )
        ]

        self.parameters = [
            (potential.parameters.detach().clone() * scale).requires_grad_()
            for potential, scale in zip(potentials, self._scales, strict=True)
        ]

    @property
    def force_field(self) -> smee.TensorForceField:
        for potential_type, parameter, scale in zip(
            self.potential_types, self.parameters, self._scales, strict=True
        ):
            potential = self._force_field.potentials_by_type[potential_type]
            potential.parameters = parameter / scale

        return self._force_field

    @torch.no_grad()
    def clamp(self):
        for parameter, constraints in zip(
            self.parameters, self._constraints, strict=True
        ):
            for i, (min_value, max_value) in constraints.items():
                if min_value is not None:
                    parameter[:, i].clamp_(min=min_value)
                if max_value is not None:
                    parameter[:, i].clamp_(max=max_value)

    @torch.no_grad()
    def freeze_grad(self):
        for parameter, col_idxs in zip(self.parameters, self._frozen_cols, strict=True):
            parameter.grad[:, col_idxs] = 0.0


@contextlib.contextmanager
def _open_writer(path: pathlib.Path | None) -> tensorboardX.SummaryWriter:
    if path is None:
        with tempfile.NamedTemporaryFile() as tmp_file:
            yield tensorboardX.SummaryWriter(tmp_file.name)

        return

    path.mkdir(parents=True, exist_ok=True)
    yield tensorboardX.SummaryWriter(str(path))


def _write_metrics(
    i: int,
    loss: torch.Tensor,
    loss_energy: torch.Tensor,
    loss_forces: torch.Tensor,
    writer: tensorboardX.SummaryWriter,
):
    writer.add_scalar("loss", loss.detach().item(), i)
    writer.add_scalar("loss_energy", loss_energy.detach().item(), i)
    writer.add_scalar("loss_forces", loss_forces.detach().item(), i)

    writer.add_scalar("rmse_energy", math.sqrt(loss_energy.detach().item()), i)
    writer.add_scalar("rmse_forces", math.sqrt(loss_forces.detach().item()), i)
    writer.flush()


def train(
    ff: smee.TensorForceField,
    top: smee.TensorTopology,
    samples: list["befit.sample.Sample"],
    n_epochs=1000,
    lr=0.01,
    params: dict[str, ParameterConfig] | None = None,
    reporter_path: pathlib.Path | None = None,
    report_interval: int = 100,
) -> smee.TensorForceField:
    """"""

    trainable = _Trainable(ff, params if params is not None else {**_DEFAULT_PARAMS})

    coords, energy, forces = [], [], []

    for sample in samples:
        coords.append(sample.coords.value_in_unit(openmm.unit.angstrom).tolist())
        energy.append(sample.energy.value_in_unit(openmm.unit.kilocalorie_per_mole))
        forces.append(
            sample.forces.value_in_unit(
                openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom
            ).tolist()
        )

    coords = torch.tensor(coords)
    energy = torch.tensor(energy)
    forces = torch.tensor(forces)

    dataset = descent.targets.energy.create_dataset(
        [{"smiles": "", "coords": coords, "energy": energy, "forces": forces}]
    )

    with _open_writer(reporter_path) as writer:
        optimizer = torch.optim.Adam(trainable.parameters, lr=lr, amsgrad=True)

        for v in tensorboardX.writer.hparams({"optimizer": "Adam", "lr": lr}, {}):
            writer.file_writer.add_summary(v)

        for i in range(n_epochs):
            e_ref, e_pred, f_ref, f_pred = descent.targets.energy.predict(
                dataset, trainable.force_field, {"": top}, "mean"
            )

            loss_energy = ((e_pred - e_ref) ** 2).sum()
            loss_forces = ((f_pred - f_ref) ** 2).sum()

            loss = loss_energy + loss_forces
            loss.backward()

            trainable.freeze_grad()

            if i % report_interval == 0:
                _write_metrics(i, loss, loss_energy, loss_forces, writer)

            optimizer.step()
            optimizer.zero_grad()

            trainable.clamp()

    return trainable.force_field
