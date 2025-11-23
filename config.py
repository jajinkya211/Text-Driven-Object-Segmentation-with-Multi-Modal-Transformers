"""
Configuration management for Referring Expression Segmentation
"""
from dataclasses import dataclass, field
from typing import Optional, List
import json
import os


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    vision_backbone: str = 'resnet101'  # 'resnet101' or 'swin'
    language_model: str = 'bert-base-uncased'

    # Vision encoder
    vision_feature_dim: int = 256
    fpn_channels: int = 256

    # Language encoder
    language_hidden_dim: int = 768
    language_output_dim: int = 256
    lstm_hidden_dim: int = 384
    lstm_num_layers: int = 2

    # Multi-modal fusion
    fusion_embed_dim: int = 256
    fusion_num_heads: int = 8
    fusion_num_layers: int = 3
    fusion_dropout: float = 0.1

    # Decoder
    decoder_channels: List[int] = field(default_factory=lambda: [128, 64, 32, 16])


@dataclass
class DataConfig:
    """Dataset configuration"""
    data_root: str = './data'
    dataset: str = 'refcoco'  # 'refcoco', 'refcoco+', 'refcocog'
    split_by: str = 'unc'

    # Image settings
    image_size: int = 480
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # Text settings
    max_text_length: int = 20

    # Data augmentation
    random_flip: bool = True
    random_crop: bool = False
    color_jitter: bool = True
    random_scale_range: tuple = (0.8, 1.2)


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 40
    accumulation_steps: int = 1

    # Scheduler
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'plateau'
    warmup_epochs: int = 2

    # Loss weights
    bce_weight: float = 1.0
    dice_weight: float = 1.0

    # Mixed precision
    use_amp: bool = True

    # Device
    device: str = 'cuda'
    num_workers: int = 4
    pin_memory: bool = True

    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    save_frequency: int = 5
    keep_best_only: bool = False

    # Logging
    log_dir: str = './logs'
    log_frequency: int = 50
    use_wandb: bool = False
    wandb_project: str = 'referring-segmentation'


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Experiment
    experiment_name: str = 'refcoco_baseline'
    seed: int = 42

    def save(self, path: str):
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'experiment_name': self.experiment_name,
            'seed': self.seed
        }

    @classmethod
    def from_dict(cls, config_dict):
        """Create from dictionary"""
        return cls(
            model=ModelConfig(**config_dict['model']),
            data=DataConfig(**config_dict['data']),
            training=TrainingConfig(**config_dict['training']),
            experiment_name=config_dict.get('experiment_name', 'refcoco_baseline'),
            seed=config_dict.get('seed', 42)
        )

    def __str__(self):
        """Pretty print configuration"""
        lines = ['Configuration:']
        lines.append(f'  Experiment: {self.experiment_name}')
        lines.append(f'  Seed: {self.seed}')
        lines.append('\n  Model:')
        for k, v in self.model.__dict__.items():
            lines.append(f'    {k}: {v}')
        lines.append('\n  Data:')
        for k, v in self.data.__dict__.items():
            lines.append(f'    {k}: {v}')
        lines.append('\n  Training:')
        for k, v in self.training.__dict__.items():
            lines.append(f'    {k}: {v}')
        return '\n'.join(lines)


def get_default_config():
    """Get default configuration"""
    return Config()


def get_config_from_args(args):
    """Create configuration from command line arguments"""
    config = Config()

    # Update from args
    if hasattr(args, 'data_root'):
        config.data.data_root = args.data_root
    if hasattr(args, 'dataset'):
        config.data.dataset = args.dataset
    if hasattr(args, 'batch_size'):
        config.training.batch_size = args.batch_size
    if hasattr(args, 'num_epochs'):
        config.training.num_epochs = args.num_epochs
    if hasattr(args, 'learning_rate'):
        config.training.learning_rate = args.learning_rate
    if hasattr(args, 'device'):
        config.training.device = args.device
    if hasattr(args, 'experiment_name'):
        config.experiment_name = args.experiment_name

    return config
