def convert_objects_to_dict(data):
    if isinstance(data, dict):
        return {key: convert_objects_to_dict(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_objects_to_dict(item) for item in data]
    elif isinstance(data, type):
        return {key: convert_objects_to_dict(value) for key, value in data.__dict__.items()
                if not key.startswith('__') and not callable(value)}
    elif hasattr(data, "__dict__"):
        return {key: convert_objects_to_dict(value) for key, value in data.__dict__.items()}
    else:
        return data
