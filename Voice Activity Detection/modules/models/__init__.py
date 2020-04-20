from .proposed import Proposed


def create_model(name, hparams):
    if name == 'Proposed':
        return Proposed(hparams)
    else:
        raise Exception('Unknown model: ' + name)
