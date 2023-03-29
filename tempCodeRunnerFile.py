my_qconfig = QConfig(
    activation=MinMaxObserver.with_args(dtype=torch.qint8),
    weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8)
    )