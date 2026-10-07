import importlib.metadata as metadata
for name in ('torch', 'tokenizers', 'transformers', 'comfy-kitchen', 'comfy-aimdo', 'pydantic', 'pydantic-settings', 'blake3', 'simpleeval', 'av', 'pytest', 'gguf', 'nvidia-ml-py'):
    try:
        print(name, metadata.version(name))
    except metadata.PackageNotFoundError:
        print(name, 'missing')
