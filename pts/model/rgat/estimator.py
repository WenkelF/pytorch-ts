from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cyclic, IterableSlice, PseudoShuffled
from gluonts.time_feature import TimeFeature
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import IterableDataset
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)
from gluonts.transform.sampler import InstanceSampler
from pts.feature.time_feature import fourier_time_features_from_frequency_str
from pts.modules import DistributionOutput
from torch.utils.data import DataLoader

from .lightning_module import RGATLightningModule
from .module import RGATModel

import wandb


PREDICTION_INPUT_NAMES = [
    "feat_static_cat",
    "feat_static_real",
    "past_time_feat",
    "past_target",
    "past_observed_values",
    "future_time_feat",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class RGATEstimator(PyTorchLightningEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        target_dim: int,
        prediction_length: int,
        context_length: Optional[int] = None,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        embedding_dim: int = 4,
        distr_output: Optional[DistributionOutput] = None,
        lin_hidden: int = 32,
        att_hidden: int = 0,
        num_att_heads: int = 8,
        activation = nn.Tanh(),
        activation_att = nn.LeakyReLU(negative_slope=0.2),
        dropout: float = 0.,
        dropout_att: float = 0.,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        scaling: bool = True,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        num_parallel_samples: int = 100,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
    ) -> None:
        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": 10.0,
        }
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)
        
        # Data
        self.freq = freq
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.cardinality = [target_dim]
        self.embedding_dim = embedding_dim

        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real

        # Model
        self.distr_output = distr_output
        self.lin_hidden = lin_hidden
        self.att_hidden = att_hidden
        self.num_att_heads = num_att_heads
        self.activation = activation
        self.activation_att = activation_att
        self.dropout = dropout
        self.dropout_att = dropout_att

        # Training
        self.lr = lr
        self.weight_decay = weight_decay
        self.scaling = scaling
        self.lags_seq = lags_seq

        self.time_features = fourier_time_features_from_frequency_str(self.freq)

        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_parallel_samples = num_parallel_samples

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

    def create_transformation(self) -> Transformation:
        remove_field_names = []
        if self.num_feat_static_real == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if self.num_feat_dynamic_real == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[i for i in range(self.target_dim)])]
                if not self.num_feat_static_cat > 0
                else []
            )
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0])]
                if not self.num_feat_static_real > 0
                else []
            )
            + [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                ),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=1 + len(self.distr_output.event_shape),
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                    log_scale=True,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.num_feat_dynamic_real > 0
                        else []
                    ),
                ),
            ]
        )

    def _create_instance_splitter(self, module: RGATLightningModule, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=module.model._past_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: RGATLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(
            module, "training"
        ) + SelectFields(TRAINING_INPUT_NAMES)

        training_instances = transformation.apply(
            Cyclic(data)
            if shuffle_buffer_length is None
            else PseudoShuffled(
                Cyclic(data), shuffle_buffer_length=shuffle_buffer_length
            )
        )

        return IterableSlice(
            iter(
                DataLoader(
                    IterableDataset(training_instances),
                    batch_size=self.batch_size,
                    **kwargs,
                    # added for avoiding reset of workers
                    persistent_workers=kwargs["num_workers"] > 0,
                )
            ),
            self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: RGATLightningModule,
        **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(
            module, "validation"
        ) + SelectFields(TRAINING_INPUT_NAMES)

        validation_instances = transformation.apply(data)

        return DataLoader(
            IterableDataset(validation_instances),
            batch_size=self.batch_size,
            **kwargs,
        )

    def create_lightning_module(self) -> RGATLightningModule:
        model = RGATModel(
            freq = self.freq,
            target_dim = self.target_dim,
            context_length = self.context_length,
            prediction_length = self.prediction_length,
            num_feat_dynamic_real = 1 + self.num_feat_dynamic_real + 2 * len(self.time_features),
            num_feat_static_cat = max(1, self.num_feat_static_cat),
            embedding_dim = self.embedding_dim,
            distr_output = self.distr_output,
            lin_hidden = self.lin_hidden,
            att_hidden = self.att_hidden,
            num_att_heads = self.num_att_heads,
            activation = self.activation,
            activation_att = self.activation_att,
            dropout = self.dropout,
            dropout_att = self.dropout_att,
            lags_seq = self.lags_seq,
            scaling = self.scaling,
            num_parallel_samples = self.num_parallel_samples
        )

        wandb.watch(model, log='gradients', log_freq=5)

        return RGATLightningModule(
            model = model,
            lr = self.lr,
            weight_decay = self.weight_decay
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: RGATLightningModule,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")

        return PyTorchPredictor(
            input_transform = transformation + prediction_splitter,
            input_names = PREDICTION_INPUT_NAMES,
            prediction_net = module.model,
            batch_size = self.batch_size,
            prediction_length = self.prediction_length,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
