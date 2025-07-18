def Convert_objects_to_dict(data):
    if isinstance(data, dict):
        return {key: Convert_objects_to_dict(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [Convert_objects_to_dict(item) for item in data]
    elif hasattr(data, "__dict__"):
        return {key: Convert_objects_to_dict(value) for key, value in data.__dict__.items()}
    else:
        return data