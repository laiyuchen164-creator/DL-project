from .baselines import DomainAdversarialVAModel, LateFusionVAModel
from .cross_modal_va import (
    AudioEncoder,
    CrossModalVAConfig,
    CrossModalVAModel,
    MViTV2SVideoBackbone,
    R2Plus1D18VideoBackbone,
    ResNet18FrameCNN,
    SharedVARegressor,
    SimpleAudioCNN,
    SimpleFrameCNN,
    TemporalEncoder,
    VideoMAEBaseVideoBackbone,
    VisualEncoder,
)
from .external_teachers import (
    DEFAULT_AUDEERING_DIM_REPO,
    DEFAULT_TEACHER_SAMPLE_RATE,
    ExternalAudeeringDimTeacher,
    TeacherCalibrationHead,
    build_external_teacher,
    extract_teacher_metadata,
)
from .paper_baselines import LeaderFollowerAttentiveFusionModel, LeaderFollowerConfig
from .losses import (
    LossWeights,
    concordance_correlation_coefficient,
    ccc_loss,
    cross_modal_training_loss,
    domain_adversarial_loss,
    regression_loss,
    symmetric_info_nce_loss,
)
