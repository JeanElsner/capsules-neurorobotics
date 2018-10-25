from .matrix_capsules import *
from .cnn import *
from .vector_capsules import *
from loss import SpreadLoss
from utils import count_parameters


def load_model(model_name, device_ids, lr, routing_iters):
    num_class = 5
    if model_name == 'matrix-capsules':
        A, B, C, D = 64, 8, 16, 16
        # A, B, C, D = 32, 32, 32, 32
        model = MatrixCapsules(A=A, B=B, C=C, D=D, E=num_class,
                               iters=routing_iters,
                               _lambda=[[1e-4, 1e-2], [1e-4, 1e-2], [1e-4, 1e-2]])
    elif model_name == 'cnn':
        model = CNN(num_class)
    elif model_name == 'vector-capsules':
        model = VectorCapsules(routing_iters, num_classes=num_class)

    model = torch.nn.DataParallel(model, device_ids=device_ids)
    print(f'Network has {count_parameters(model):d} parameters')

    criterion = SpreadLoss(num_class=num_class, m_min=0.2, m_max=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)
    return model, criterion, optimizer, scheduler
