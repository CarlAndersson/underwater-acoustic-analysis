import functools
import inspect


def prebind(func, **prebound_kwargs):
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs = prebound_kwargs | kwargs
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        if set(bound.arguments) == set(sig.parameters):
            return func(*args, **kwargs)
        else:
            wrapped = prebind(func, **bound.arguments)
            # Documentation of pre-bound values for human use
            if wrapped.__doc__:
                bound_docs = ['\n\nPre-bound parameters', '-' * 20]
                for key, value in bound.arguments.items():
                    bound_docs.append(f'{key}: {value}')
                wrapped.__doc__ += '\n'.join(bound_docs)
            # "Documentation" of pre-bound values for programmatic use
            wrapped._pre_bound = bound.arguments
            return wrapped

    return wrapper
